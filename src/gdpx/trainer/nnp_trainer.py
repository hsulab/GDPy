import copy
import json
import time
from collections.abc import Mapping

import numpy as np
from ase.io import write

from .trainer import BasePotentialTrainer

WEIGHTS_NAME = "nn_weights.npz"
INPUT_DATASET_NAME = "input_dataset.xyz"


class NnpTrainer(BasePotentialTrainer):
    name = "nnp"

    def __init__(
        self,
        config,
        type_list=None,
        train_epochs=1000,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed=None,
        calculator_params=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            type_list=type_list,
            train_epochs=train_epochs,
            directory=directory,
            command=command,
            freeze_command=freeze_command,
            random_seed=random_seed,
        )
        self.calculator_params = calculator_params or {}

    @property
    def frozen_name(self):
        return WEIGHTS_NAME

    def _resolve_train_command(self, *args, **kwargs):
        return ""

    def _resolve_freeze_command(self, *args, **kwargs):
        return ""

    def write_input(self, dataset, *args, **kwargs):
        if dataset:
            write(self.directory / INPUT_DATASET_NAME, dataset)

    def read_convergence(self) -> bool:
        return True

    def train(self, dataset, init_model=None, *args, **kwargs):
        train_dir = self.directory
        train_dir.mkdir(parents=True, exist_ok=True)

        dataset = list(dataset)
        if not dataset:
            raise ValueError("NnpTrainer requires a non-empty dataset.")

        self.write_input(dataset)

        calculator_params = self.calculator_params
        train_config = self.config
        n_epochs = train_config.get("n_epochs", self.train_epochs)
        legacy_loss_keys = sorted(
            {"energy_weight", "force_weight", "gradient_balance"}
            & set(train_config)
        )
        if legacy_loss_keys:
            raise ValueError(
                "Legacy NNP loss options are no longer supported: "
                f"{legacy_loss_keys}. Replace them with `learning_rate: "
                "{start, stop}` and `loss: {start_pref_e, limit_pref_e, "
                "start_pref_f, limit_pref_f}`."
            )

        learning_rate_config = train_config.get("learning_rate", {})
        if not isinstance(learning_rate_config, Mapping):
            raise ValueError(
                "`learning_rate` must be a mapping with `start` and `stop`; "
                "scalar learning rates belong to the removed NNP loss mode."
            )
        start_learning_rate = float(learning_rate_config.get("start", 0.003))
        stop_learning_rate = float(learning_rate_config.get("stop", 1.0e-6))

        loss_config = train_config.get("loss", {})
        if not isinstance(loss_config, Mapping):
            raise ValueError("`loss` must be a mapping of DeepMD-style prefactors.")
        start_pref_e = float(loss_config.get("start_pref_e", 0.02))
        limit_pref_e = float(loss_config.get("limit_pref_e", 1.0))
        start_pref_f = float(loss_config.get("start_pref_f", 1000.0))
        limit_pref_f = float(loss_config.get("limit_pref_f", 1.0))
        train_forces = start_pref_f > 0.0 or limit_pref_f > 0.0
        verbose = train_config.get("verbose", 100)
        max_grad_norm = train_config.get("max_grad_norm", None)
        batch_size = train_config.get("batch_size", None)
        shuffle = bool(train_config.get("shuffle", True))
        validation_fraction = float(train_config.get("validation_fraction", 0.0))
        early_stopping_patience = train_config.get("early_stopping_patience", None)
        normalization_epsilon = float(
            train_config.get("normalization_epsilon", 1.0e-8)
        )

        if n_epochs < 1:
            raise ValueError(f"n_epochs must be positive; got {n_epochs}.")
        if start_learning_rate <= 0.0 or stop_learning_rate <= 0.0:
            raise ValueError(
                "learning-rate endpoints must be positive; got "
                f"start={start_learning_rate}, stop={stop_learning_rate}."
            )
        if stop_learning_rate > start_learning_rate:
            raise ValueError(
                "learning_rate.stop must not exceed learning_rate.start."
            )
        prefactors = (start_pref_e, limit_pref_e, start_pref_f, limit_pref_f)
        if any(prefactor < 0.0 for prefactor in prefactors):
            raise ValueError("All DeepMD loss prefactors must be non-negative.")
        if not any(prefactors):
            raise ValueError("At least one DeepMD loss prefactor must be positive.")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be in [0, 1); got "
                f"{validation_fraction}."
            )
        if early_stopping_patience is not None:
            early_stopping_patience = int(early_stopping_patience)
            if early_stopping_patience < 1:
                raise ValueError("early_stopping_patience must be positive.")

        if not calculator_params:
            raise ValueError(
                "NnpTrainer requires `calculator_params` in the trainer config with keys: "
                "elements, g2_params, g4_params, r_cut, hidden_sizes."
            )

        elements = calculator_params["elements"]
        g2_params_raw = calculator_params["g2_params"]
        g4_params_raw = calculator_params.get("g4_params", [])
        r_cut = calculator_params["r_cut"]
        hidden_sizes = calculator_params.get("hidden_sizes", (64, 64))

        from gdpx.potential.nnp.descriptor import (
            G2Param,
            G4Param,
            compute_n_features,
            compute_symmetry_functions_and_derivatives,
        )

        g2_norm = [G2Param(*p) if not isinstance(p, G2Param) else p for p in g2_params_raw]
        g4_norm = [G4Param(*p) if not isinstance(p, G4Param) else p for p in g4_params_raw]
        n_features = compute_n_features(elements, g2_norm, g4_norm)

        ref_energies = np.zeros(len(dataset))
        symbols = []
        for idx, atoms in enumerate(dataset):
            if atoms.calc is None or "energy" not in atoms.calc.results:
                raise ValueError(
                    f"Structure {idx} has no reference energy "
                    "(atoms.get_potential_energy() unavailable). "
                    "Provide reference energies in the dataset."
                )
            ref_energies[idx] = atoms.get_potential_energy()
            if train_forces and (
                atoms.calc is None or "forces" not in atoms.calc.results
            ):
                raise ValueError(
                    f"Structure {idx} has no reference forces "
                    "(atoms.get_forces(apply_constraint=False) unavailable); "
                    "required because the force prefactors are nonzero."
                )
            current_symbols = np.asarray(atoms.get_chemical_symbols(), dtype=str)
            unknown = sorted(set(current_symbols) - set(elements))
            if unknown:
                raise ValueError(
                    f"Structure {idx} contains elements {unknown} not present "
                    f"in calculator elements {elements}."
                )
            symbols.append(current_symbols)

        cache_start = time.perf_counter()
        cached = [
            compute_symmetry_functions_and_derivatives(
                atoms, elements, g2_norm, g4_norm, r_cut
            )
            for atoms in dataset
        ]
        cache_seconds = time.perf_counter() - cache_start
        cache_bytes = sum(
            descriptor.nbytes + jacobian.nbytes
            for descriptor, jacobian in cached
        )

        indices = self.rng.permutation(len(dataset))
        n_validation = int(round(validation_fraction * len(dataset)))
        if validation_fraction > 0.0:
            n_validation = max(1, n_validation)
            if n_validation >= len(dataset):
                raise ValueError(
                    "validation_fraction leaves no structures for training."
                )
        validation_indices = indices[:n_validation]
        training_indices = indices[n_validation:]

        feature_mean = np.zeros((len(elements), n_features), dtype=float)
        feature_scale = np.ones((len(elements), n_features), dtype=float)
        for element_index, element in enumerate(elements):
            environments = [
                cached[index][0][symbols[index] == element]
                for index in training_indices
            ]
            environments = [values for values in environments if len(values)]
            if not environments:
                raise ValueError(
                    f"No training environments are available for element {element}."
                )
            values = np.concatenate(environments, axis=0)
            feature_mean[element_index] = np.mean(values, axis=0)
            standard_deviation = np.std(values, axis=0)
            feature_scale[element_index] = np.where(
                standard_deviation > normalization_epsilon,
                standard_deviation,
                1.0,
            )

        composition = np.array(
            [
                [np.count_nonzero(symbols[index] == element) for element in elements]
                for index in training_indices
            ],
            dtype=float,
        )
        atomic_offsets = np.linalg.lstsq(
            composition, ref_energies[training_indices], rcond=None
        )[0]

        from gdpx.potential.nnp.nn import ElementwiseNN

        model = ElementwiseNN(
            n_features,
            elements,
            hidden_sizes=hidden_sizes,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            atomic_offsets=atomic_offsets,
            rng=self.rng,
        )
        if init_model is not None:
            from gdpx.potential.nnp.calculator import ACSFNN

            initial = ACSFNN(init_model)
            _validate_initial_model(model, initial, calculator_params)
            model.feature_mean[...] = initial.model.feature_mean
            model.feature_scale[...] = initial.model.feature_scale
            model.atomic_offsets[...] = initial.model.atomic_offsets
            model.set_params(initial.model.get_params())

        if batch_size is None:
            batch_size = len(training_indices)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        batch_size = min(batch_size, len(training_indices))
        batches_per_epoch = (
            len(training_indices) + batch_size - 1
        ) // batch_size
        total_training_steps = n_epochs * batches_per_epoch

        config_dump = dict(
            name=self.name,
            model_format_version=2,
            training=dict(
                n_epochs=n_epochs,
                learning_rate=dict(
                    start=start_learning_rate,
                    stop=stop_learning_rate,
                    schedule="smooth_exponential",
                ),
                loss=dict(
                    type="deepmd",
                    start_pref_e=start_pref_e,
                    limit_pref_e=limit_pref_e,
                    start_pref_f=start_pref_f,
                    limit_pref_f=limit_pref_f,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                max_grad_norm=max_grad_norm,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
            ),
            dataset=dict(
                n_structures=len(dataset),
                n_training=len(training_indices),
                n_validation=len(validation_indices),
                max_atoms=max(len(a) for a in dataset),
                min_atoms=min(len(a) for a in dataset),
                cache_seconds=cache_seconds,
                cache_bytes=cache_bytes,
            ),
            calculator=copy.deepcopy(calculator_params),
        )
        with open(train_dir / "train_config.json", "w") as f:
            json.dump(config_dump, f, indent=2)

        self._print(
            f"Cached {len(dataset)} descriptor Jacobians in "
            f"{cache_seconds:.3f} s ({cache_bytes / 1024**2:.2f} MiB)."
        )

        adam_state = None
        history = []
        best_score = np.inf
        best_params = None
        epochs_without_improvement = 0
        global_step = 0
        log_header_printed = False

        for epoch in range(n_epochs):
            epoch_start = time.perf_counter()
            epoch_indices = np.array(training_indices, copy=True)
            if shuffle:
                self.rng.shuffle(epoch_indices)
            energy_gradient_norms = []
            force_gradient_norms = []
            effective_energy_gradient_norms = []
            effective_force_gradient_norms = []
            gradient_cosines = []
            epoch_learning_rates = []
            energy_prefactors = []
            force_prefactors = []

            for batch_start in range(0, len(epoch_indices), batch_size):
                batch = epoch_indices[batch_start : batch_start + batch_size]
                energy_grads = None
                force_grads = None
                learning_rate = _smooth_exponential_learning_rate(
                    global_step,
                    total_training_steps,
                    start_learning_rate,
                    stop_learning_rate,
                )
                energy_prefactor = _scheduled_prefactor(
                    learning_rate,
                    start_learning_rate,
                    start_pref_e,
                    limit_pref_e,
                )
                force_prefactor = _scheduled_prefactor(
                    learning_rate,
                    start_learning_rate,
                    start_pref_f,
                    limit_pref_f,
                )
                epoch_learning_rates.append(learning_rate)
                energy_prefactors.append(energy_prefactor)
                force_prefactors.append(force_prefactor)

                for index in batch:
                    atoms = dataset[index]
                    descriptor, jacobian = cached[index]
                    atom_symbols = symbols[index]
                    num_atoms = len(atoms)

                    energy_pred = float(
                        np.sum(model.forward(descriptor, atom_symbols))
                    )
                    energy_error = energy_pred - ref_energies[index]
                    current = model.backward(
                        np.full(
                            num_atoms,
                            _deepmd_energy_output_gradient(
                                energy_error, num_atoms, len(batch)
                            ),
                        )
                    )
                    energy_grads = _accumulate_elementwise_grads(
                        energy_grads, current
                    )

                    if train_forces:
                        _, dE_dG = model.energy_and_gradient(
                            descriptor, atom_symbols
                        )
                        force_error = (
                            jacobian.forces(dE_dG)
                            - atoms.get_forces(apply_constraint=False)
                        )
                        current = model.double_backward(
                            jacobian.adjoint(force_error)
                        )
                        _scale_elementwise_grads(
                            current,
                            _force_double_backward_scale(
                                num_atoms, len(batch)
                            ),
                        )
                        force_grads = _accumulate_elementwise_grads(
                            force_grads, current
                        )

                energy_norm = _elementwise_global_norm(energy_grads)
                force_norm = _elementwise_global_norm(force_grads)
                energy_gradient_norms.append(energy_norm)
                force_gradient_norms.append(force_norm)
                gradient_cosines.append(
                    _elementwise_cosine(energy_grads, force_grads)
                )
                effective_energy_gradient_norms.append(
                    energy_prefactor * energy_norm
                )
                effective_force_gradient_norms.append(
                    force_prefactor * force_norm
                )
                _scale_elementwise_grads(energy_grads, energy_prefactor)
                _scale_elementwise_grads(force_grads, force_prefactor)
                total_grads = _accumulate_elementwise_grads(
                    energy_grads, force_grads
                )

                if max_grad_norm is not None:
                    grad_norm = _elementwise_global_norm(total_grads)
                    if grad_norm > max_grad_norm:
                        _scale_elementwise_grads(
                            total_grads, max_grad_norm / grad_norm
                        )
                adam_state = model.adam_update(
                    total_grads, learning_rate, adam_state
                )
                global_step += 1

            train_metrics = _evaluate_model(
                model,
                dataset,
                symbols,
                ref_energies,
                cached,
                training_indices,
                train_forces,
            )
            validation_metrics = (
                _evaluate_model(
                    model,
                    dataset,
                    symbols,
                    ref_energies,
                    cached,
                    validation_indices,
                    train_forces,
                )
                if len(validation_indices)
                else None
            )
            selection_metrics = validation_metrics or train_metrics
            score = _fixed_selection_score(
                selection_metrics, limit_pref_e, limit_pref_f
            )
            improved = score < best_score
            if improved:
                best_score = score
                best_params = model.get_params()
                epochs_without_improvement = 0
                _save_model(
                    train_dir / WEIGHTS_NAME,
                    model,
                    hidden_sizes,
                    elements,
                    g2_norm,
                    g4_norm,
                    r_cut,
                )
            else:
                epochs_without_improvement += 1

            row = dict(
                epoch=epoch,
                train_energy_rmse=train_metrics["energy_rmse"],
                train_force_rmse=train_metrics["force_rmse"],
                train_energy_loss=train_metrics["energy_loss"],
                train_force_loss=train_metrics["force_loss"],
                train_total_loss=(
                    float(np.mean(energy_prefactors))
                    * train_metrics["energy_loss"]
                    + float(np.mean(force_prefactors))
                    * train_metrics["force_loss"]
                ),
                validation_energy_rmse=(
                    validation_metrics["energy_rmse"]
                    if validation_metrics is not None
                    else None
                ),
                validation_force_rmse=(
                    validation_metrics["force_rmse"]
                    if validation_metrics is not None
                    else None
                ),
                validation_energy_loss=(
                    validation_metrics["energy_loss"]
                    if validation_metrics is not None
                    else None
                ),
                validation_force_loss=(
                    validation_metrics["force_loss"]
                    if validation_metrics is not None
                    else None
                ),
                selection_score=score,
                energy_gradient_norm=float(np.mean(energy_gradient_norms)),
                force_gradient_norm=float(np.mean(force_gradient_norms)),
                effective_energy_gradient_norm=float(
                    np.mean(effective_energy_gradient_norms)
                ),
                effective_force_gradient_norm=float(
                    np.mean(effective_force_gradient_norms)
                ),
                gradient_cosine=float(np.mean(gradient_cosines)),
                learning_rate=float(np.mean(epoch_learning_rates)),
                energy_prefactor=float(np.mean(energy_prefactors)),
                force_prefactor=float(np.mean(force_prefactors)),
                epoch_seconds=time.perf_counter() - epoch_start,
                best=improved,
            )
            history.append(row)
            with open(train_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

            if verbose and (epoch % verbose == 0 or epoch == n_epochs - 1):
                has_validation = validation_metrics is not None
                if not log_header_printed:
                    self._print(_training_log_header(has_validation))
                    log_header_printed = True
                self._print(_training_log_row(row, has_validation))

            if (
                early_stopping_patience is not None
                and len(validation_indices)
                and epochs_without_improvement >= early_stopping_patience
            ):
                self._print(
                    f"Early stopping at epoch {epoch}; validation score did "
                    f"not improve for {early_stopping_patience} epochs."
                )
                break

        if best_params is not None:
            model.set_params(best_params)
        _save_model(
            train_dir / WEIGHTS_NAME,
            model,
            hidden_sizes,
            elements,
            g2_norm,
            g4_norm,
            r_cut,
        )

    def freeze(self):
        model_path = (self.directory / self.frozen_name).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Trained weights not found at {model_path}. Run train() first.")
        return model_path


def _global_norm(grads):
    total = 0.0
    for arr_list in grads.values():
        for arr in arr_list:
            total += np.sum(arr**2)
    return np.sqrt(total)


def _copy_grads(grads):
    return {
        "weights": [w.copy() for w in grads["weights"]],
        "biases": [b.copy() for b in grads["biases"]],
    }


def _add_grads(target, source):
    for i in range(len(target["weights"])):
        target["weights"][i] += source["weights"][i]
    for i in range(len(target["biases"])):
        target["biases"][i] += source["biases"][i]


def _scale_grads(grads, factor):
    for i in range(len(grads["weights"])):
        grads["weights"][i] *= factor
    for i in range(len(grads["biases"])):
        grads["biases"][i] *= factor


def _copy_elementwise_grads(grads):
    if grads is None:
        return None
    return {element: _copy_grads(values) for element, values in grads.items()}


def _accumulate_elementwise_grads(target, source):
    if source is None:
        return target
    if target is None:
        return _copy_elementwise_grads(source)
    for element in target:
        _add_grads(target[element], source[element])
    return target


def _scale_elementwise_grads(grads, factor):
    if grads is None:
        return
    for values in grads.values():
        _scale_grads(values, factor)


def _elementwise_global_norm(grads):
    if grads is None:
        return 0.0
    return float(
        np.sqrt(sum(_global_norm(values) ** 2 for values in grads.values()))
    )


def _elementwise_dot(first, second):
    if first is None or second is None:
        return 0.0
    total = 0.0
    for element in first:
        for first_array, second_array in zip(
            first[element]["weights"], second[element]["weights"]
        ):
            total += float(np.vdot(first_array, second_array))
        for first_array, second_array in zip(
            first[element]["biases"], second[element]["biases"]
        ):
            total += float(np.vdot(first_array, second_array))
    return total


def _elementwise_cosine(first, second):
    first_norm = _elementwise_global_norm(first)
    second_norm = _elementwise_global_norm(second)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    cosine = _elementwise_dot(first, second) / (first_norm * second_norm)
    return float(np.clip(cosine, -1.0, 1.0))


def _smooth_exponential_learning_rate(
    step, total_steps, start_learning_rate, stop_learning_rate
):
    """Return a smooth exponential schedule including both endpoints."""
    if total_steps <= 1:
        return float(start_learning_rate)
    progress = np.clip(step / (total_steps - 1), 0.0, 1.0)
    return float(
        start_learning_rate
        * (stop_learning_rate / start_learning_rate) ** progress
    )


def _scheduled_prefactor(
    learning_rate,
    start_learning_rate,
    start_prefactor,
    limit_prefactor,
):
    ratio = learning_rate / start_learning_rate
    return float(start_prefactor * ratio + limit_prefactor * (1.0 - ratio))


def _deepmd_energy_loss(energy_error, num_atoms):
    return float(energy_error**2 / num_atoms)


def _deepmd_energy_output_gradient(energy_error, num_atoms, batch_size):
    return float(2.0 * energy_error / (batch_size * num_atoms))


def _force_double_backward_scale(num_atoms, batch_size):
    # Forces contain a minus derivative of energy, hence the negative sign.
    return float(-2.0 / (batch_size * 3 * num_atoms))


def _fixed_selection_score(metrics, energy_prefactor, force_prefactor):
    return float(
        energy_prefactor * metrics["energy_loss"]
        + force_prefactor * metrics["force_loss"]
    )


def _training_log_header(has_validation):
    columns = ["epoch", "E_tr/atom", "F_tr"]
    if has_validation:
        columns.extend(["E_val/atom", "F_val"])
    columns.extend(["lr", "sec"])
    return f"{columns[0]:>6s}" + "".join(
        f" {column:>11s}" for column in columns[1:]
    )


def _training_log_row(row, has_validation):
    values = [row["train_energy_rmse"], row["train_force_rmse"]]
    if has_validation:
        values.extend(
            [row["validation_energy_rmse"], row["validation_force_rmse"]]
        )
    values.extend(
        [
            row["learning_rate"],
            row["epoch_seconds"],
        ]
    )
    return f"{row['epoch']:6d}" + "".join(
        f" {value:11.4e}" for value in values
    )


def _evaluate_model(
    model,
    dataset,
    symbols,
    ref_energies,
    cached,
    indices,
    evaluate_forces,
):
    energy_squared = 0.0
    energy_loss = 0.0
    force_squared = 0.0
    force_components = 0
    force_loss = 0.0
    for index in indices:
        descriptor, jacobian = cached[index]
        atomic_energies, dE_dG = model.energy_and_gradient(
            descriptor, symbols[index]
        )
        energy_error = float(np.sum(atomic_energies)) - ref_energies[index]
        energy_error_per_atom = energy_error / len(dataset[index])
        energy_squared += energy_error_per_atom**2
        energy_loss += _deepmd_energy_loss(
            energy_error, len(dataset[index])
        )
        if evaluate_forces:
            force_error = (
                jacobian.forces(dE_dG)
                - dataset[index].get_forces(apply_constraint=False)
            )
            force_squared += float(np.sum(force_error**2))
            force_components += force_error.size
            force_loss += float(np.mean(force_error**2))
    return {
        "energy_rmse": float(np.sqrt(energy_squared / len(indices))),
        "force_rmse": (
            float(np.sqrt(force_squared / force_components))
            if force_components
            else 0.0
        ),
        "energy_loss": float(energy_loss / len(indices)),
        "force_loss": (
            float(force_loss / len(indices)) if evaluate_forces else 0.0
        ),
    }


def _save_model(
    model_path,
    model,
    hidden_sizes,
    elements,
    g2_params,
    g4_params,
    r_cut,
):
    save_dict = {
        "format_version": np.int64(2),
        "hidden_sizes": np.asarray(hidden_sizes, dtype=np.int64),
        "elements": np.asarray(elements),
        "feature_mean": model.feature_mean,
        "feature_scale": model.feature_scale,
        "atomic_offsets": model.atomic_offsets,
        "g2_eta": np.array([parameter.eta for parameter in g2_params]),
        "g2_Rs": np.array([parameter.Rs for parameter in g2_params]),
        "g4_eta": np.array([parameter.eta for parameter in g4_params]),
        "g4_zeta": np.array([parameter.zeta for parameter in g4_params]),
        "g4_lambda_": np.array([parameter.lambda_ for parameter in g4_params]),
        "r_cut": np.float64(r_cut),
    }
    model.save_parameters(save_dict)
    np.savez_compressed(model_path, **save_dict)


def _validate_initial_model(model, initial, calculator_params):
    initial_model = initial.model
    if model.elements != initial_model.elements:
        raise ValueError(
            "Initial model elements do not match the requested calculator "
            f"elements: {initial_model.elements} != {model.elements}."
        )
    if model.n_input != initial_model.n_input:
        raise ValueError(
            "Initial model descriptor width does not match the requested "
            f"configuration: {initial_model.n_input} != {model.n_input}."
        )
    if model.hidden_sizes != initial_model.hidden_sizes:
        raise ValueError(
            "Initial model hidden sizes do not match the requested "
            f"configuration: {initial_model.hidden_sizes} != "
            f"{model.hidden_sizes}."
        )
    expected_g2 = np.asarray(calculator_params["g2_params"], dtype=float)
    actual_g2 = np.asarray(
        [(parameter.eta, parameter.Rs) for parameter in initial.g2_params],
        dtype=float,
    ).reshape((-1, 2))
    expected_g4 = np.asarray(
        calculator_params.get("g4_params", []), dtype=float
    ).reshape((-1, 3))
    actual_g4 = np.asarray(
        [
            (parameter.eta, parameter.zeta, parameter.lambda_)
            for parameter in initial.g4_params
        ],
        dtype=float,
    ).reshape((-1, 3))
    if (
        expected_g2.shape != actual_g2.shape
        or not np.allclose(expected_g2, actual_g2)
        or expected_g4.shape != actual_g4.shape
        or not np.allclose(expected_g4, actual_g4)
        or not np.isclose(float(calculator_params["r_cut"]), initial.r_cut)
    ):
        raise ValueError(
            "Initial model descriptor parameters do not match the requested "
            "calculator configuration."
        )
