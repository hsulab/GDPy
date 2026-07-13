import numpy as np

from .fgp import FGP
from .sgp import SGP


def save_model(gp, path):
    state = {
        "class_name": type(gp).__name__,
        "hypers": gp._hypers,
        "r_cut_2b": gp.r_cut_2b,
        "r_cut_3b": gp.r_cut_3b,
        "jitter": gp.jitter,
        "use_2b": gp.use_2b,
        "use_3b": gp.use_3b,
        "inducing_2b": gp.inducing_2b,
        "inducing_3b": gp.inducing_3b,
        "_Kmm": gp._Kmm,
        "_Knm": gp._Knm,
        "L": gp.L,
        "alpha": gp.alpha,
        "nframes": gp.nframes,
        "natoms_tot": gp.natoms_tot,
    }

    for key in ["body2_features", "body2_gradients", "body2_mapping", "body2_species",
                 "body3_features", "body3_gradients", "body3_mapping", "body3_species"]:
        state[key] = gp.desc.get(key, np.array([]))

    for attr in ["_species_2b_order", "_species_3b_order"]:
        val = getattr(gp, attr, None)
        state[attr] = np.array(val, dtype=object) if val else np.array([])

    if hasattr(gp, "n_inducing"):
        state["n_inducing"] = gp.n_inducing

    np.savez_compressed(path, **state)


def load_model(path):
    data = np.load(path, allow_pickle=True)
    cls_name = str(data["class_name"])

    hypers = data["hypers"]
    use_2b = bool(data["use_2b"])
    use_3b = bool(data["use_3b"])
    r_cut_2b = float(data["r_cut_2b"])
    r_cut_3b = float(data["r_cut_3b"])
    jitter = float(data["jitter"])

    common = dict(
        r_cut_2b=r_cut_2b, r_cut_3b=r_cut_3b,
        sigma_2b=hypers[0], length_2b=hypers[1],
        sigma_3b=hypers[2], length_3b=hypers[3],
        noise=hypers[4], jitter=jitter,
        use_2b=use_2b, use_3b=use_3b,
    )

    if cls_name == "FGP":
        gp = FGP(**common)
    elif cls_name == "SGP":
        n_inducing = int(data.get("n_inducing", 100))
        gp = SGP(n_inducing=n_inducing, **common)
    else:
        raise ValueError(f"Unknown class: {cls_name}")

    gp.inducing_2b = data["inducing_2b"]
    gp.inducing_3b = data["inducing_3b"]
    gp._Kmm = data["_Kmm"]
    gp._Knm = data["_Knm"]
    gp.L = data["L"]
    gp.alpha = data["alpha"]
    gp.nframes = int(data["nframes"])
    gp.natoms_tot = int(data["natoms_tot"])

    gp.desc = {}
    for key in ["body2_features", "body2_gradients", "body2_mapping", "body2_species",
                 "body3_features", "body3_gradients", "body3_mapping", "body3_species"]:
        arr = data.get(key)
        if arr is not None and arr.size > 0:
            gp.desc[key] = arr

    so2b = data.get("_species_2b_order")
    if so2b is not None and so2b.size > 0:
        gp._species_2b_order = list(so2b)
    so3b = data.get("_species_3b_order")
    if so3b is not None and so3b.size > 0:
        gp._species_3b_order = list(so3b)

    return gp
