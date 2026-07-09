import numpy as np
from ase.io import read


def test_fgp_fit_predict():
    from gdpx.potential.gp import FGP

    traj = read("test_data/cu13_md.traj@:")
    forces = np.load("test_data/cu13_forces.npy")

    gp = FGP(r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5, noise=0.1,
             jitter=1e-3, use_2b=True, use_3b=False)
    gp.fit(traj[:3], list(forces[:3]))

    pred, std = gp.predict(traj[3:5], return_std=True)
    true_f = forces[3:5].ravel()

    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(std))
    assert std.min() > 0
    assert np.corrcoef(pred, true_f)[0, 1] > 0.9


def test_sgp_fit_predict():
    from gdpx.potential.gp import SGP

    traj = read("test_data/cu13_md.traj@:")
    forces = np.load("test_data/cu13_forces.npy")

    gp = SGP(n_inducing=50, r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5,
             noise=0.1, jitter=1e-3, use_2b=True, use_3b=False)
    gp.fit(traj[:5], list(forces[:5]))

    pred, std = gp.predict(traj[5:10], return_std=True)
    true_f = forces[5:10].ravel()

    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(std))
    assert np.corrcoef(pred, true_f)[0, 1] > 0.9


def test_gp_optimize():
    from gdpx.potential.gp import SGP

    traj = read("test_data/cu13_md.traj@:")
    forces = np.load("test_data/cu13_forces.npy")

    gp = SGP(n_inducing=50, r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5,
             noise=0.1, jitter=1e-3, use_2b=True, use_3b=False)
    gp.fit(traj[:4], list(forces[:4]))

    lml_before = gp.log_marginal_likelihood(gp.hypers)
    result = gp.optimize(maxiter=30)
    lml_after = gp.log_marginal_likelihood(gp.hypers)

    assert result.success
    assert lml_after >= lml_before - 1e-6


def test_fgp_in_sample_accuracy():
    from gdpx.potential.gp import FGP

    traj = read("test_data/cu13_md.traj@:")
    forces = np.load("test_data/cu13_forces.npy")

    gp = FGP(r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5, noise=0.1,
             jitter=1e-2, use_2b=True, use_3b=False)
    gp.fit(traj[:3], list(forces[:3]))

    pred, _ = gp.predict(traj[:3], return_std=True)
    true_f = forces[:3].ravel()

    corr = np.corrcoef(pred, true_f)[0, 1]
    assert corr > 0.95, f"FGP in-sample corr {corr:.4f} < 0.95"


def test_sgp_in_sample_accuracy():
    from gdpx.potential.gp import SGP

    traj = read("test_data/cu13_md.traj@:")
    forces = np.load("test_data/cu13_forces.npy")

    gp = SGP(n_inducing=80, r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5,
             noise=0.1, jitter=1e-3, use_2b=True, use_3b=False)
    gp.fit(traj[:3], list(forces[:3]))

    pred, _ = gp.predict(traj[:3], return_std=True)
    true_f = forces[:3].ravel()

    corr = np.corrcoef(pred, true_f)[0, 1]
    assert corr > 0.95, f"SGP in-sample corr {corr:.4f} < 0.95"


def test_sgp_periodic_bulk():
    from gdpx.potential.gp import SGP

    traj = read("test_data/cu32_md.traj@:")
    forces = np.load("test_data/cu32_forces.npy")

    gp = SGP(n_inducing=100, r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.3,
             noise=0.1, jitter=1e-2, use_2b=True, use_3b=False)
    gp.fit(traj[:5], list(forces[:5]))

    pred, std = gp.predict(traj[5:8], return_std=True)
    true_f = forces[5:8].ravel()

    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(std))
    corr = np.corrcoef(pred, true_f)[0, 1]
    assert corr > 0.85, f"periodic bulk corr {corr:.4f} < 0.85"


def test_sgp_multielement_aucu():
    from gdpx.potential.gp import SGP

    traj = read("test_data/aucu_md.traj@:")
    forces = np.load("test_data/aucu_forces.npy")

    gp = SGP(n_inducing=100, r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5,
             noise=0.1, jitter=1e-2, use_2b=True, use_3b=False)
    gp.fit(traj[:5], list(forces[:5]))

    pred, std = gp.predict(traj[5:10], return_std=True)
    true_f = forces[5:10].ravel()

    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(std))
    corr = np.corrcoef(pred, true_f)[0, 1]
    assert corr > 0.85, f"AuCu multi-element corr {corr:.4f} < 0.85"


def test_fgp_multielement_aucu():
    from gdpx.potential.gp import FGP

    traj = read("test_data/aucu_md.traj@:")
    forces = np.load("test_data/aucu_forces.npy")

    gp = FGP(r_cut_2b=6.0, sigma_2b=1.0, length_2b=0.5, noise=0.1,
             jitter=1e-2, use_2b=True, use_3b=False)
    gp.fit(traj[:3], list(forces[:3]))

    pred, _ = gp.predict(traj[:3], return_std=True)
    true_f = forces[:3].ravel()
    corr = np.corrcoef(pred, true_f)[0, 1]
    assert corr > 0.9, f"FGP AuCu in-sample corr {corr:.4f} < 0.9"
