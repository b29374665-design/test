"""Equivalent-circuit fit and Monte Carlo uncertainty.

Circuit:  L - R0 - (R1 || CPE1) - (R2 || CPE2) - W
  R0   ohmic resistance (electrolyte, contacts, current collectors)
  R1   high-frequency arc (SEI / surface film)
  R2   mid-frequency arc (charge transfer)
  W    semi-infinite Warburg diffusion tail
"""
import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ["L", "R0", "R1", "Q1", "a1", "R2", "Q2", "a2", "Aw"]
_LOG = [True, True, True, True, False, True, True, False, True]
_LB = [1e-12, 1e-7, 1e-7, 1e-9, 0.5, 1e-7, 1e-9, 0.5, 1e-9]
_UB = [1e-4, 1e3, 1e3, 1e5, 1.0, 1e3, 1e5, 1.0, 1e3]


def impedance(p, omega):
    L, R0, R1, Q1, a1, R2, Q2, a2, Aw = p
    jw = 1j * omega
    arc1 = R1 / (1 + R1 * Q1 * jw ** a1)
    arc2 = R2 / (1 + R2 * Q2 * jw ** a2)
    return jw * L + R0 + arc1 + arc2 + Aw / np.sqrt(jw)


def _to_x(p):
    return np.array([np.log(v) if lg else v for v, lg in zip(p, _LOG)])


def _from_x(x):
    return np.array([np.exp(v) if lg else v for v, lg in zip(x, _LOG)])


_XLB = _to_x(_LB)
_XUB = _to_x(_UB)


def _residual(x, omega, z):
    zm = impedance(_from_x(x), omega)
    d = (zm - z) / np.abs(z)
    return np.concatenate([d.real, d.imag])


def model_free_features(spec):
    """Graphical estimates straight from the Nyquist plot."""
    z, f = spec.z, spec.freq
    nim = -z.imag
    # R0: high-frequency real-axis intercept (where -Im crosses zero).
    cross = np.where(np.diff(np.sign(nim)) > 0)[0]
    if len(cross):
        i = cross[0]
        t = nim[i] / (nim[i] - nim[i + 1])
        r0 = z.real[i] + t * (z.real[i + 1] - z.real[i])
    else:
        r0 = z.real[np.argmin(np.abs(nim))] if nim.min() <= 0 else z.real.min()
    # End of the arcs: local minimum of -Im before the diffusion tail rises.
    cap = np.where(nim > 0)[0]
    arc_end = None
    if len(cap) > 3:
        seg = nim[cap]
        for k in range(1, len(seg) - 1):
            if seg[k] < seg[k - 1] and seg[k] <= seg[k + 1] and seg[:k].max() > seg[k] * 1.05:
                arc_end = cap[k]
        # keep the last minimum (after all arcs)
    rp = (z.real[arc_end] if arc_end is not None else z.real.max()) - r0
    peak = cap[np.argmax(nim[cap])] if len(cap) else 0
    return {"R0": float(r0), "Rp": float(max(rp, 1e-6)), "f_peak": float(f[peak]),
            "has_tail": arc_end is not None}


def _initial_guesses(spec):
    feat = model_free_features(spec)
    r0, rp, fp = feat["R0"], feat["Rp"], feat["f_peak"]
    tau2 = 1 / (2 * np.pi * fp)
    lowest = spec.z[-1]
    aw0 = max(abs(lowest.imag) * np.sqrt(spec.omega[-1]) / np.sqrt(2), 1e-6) if feat["has_tail"] else 1e-6
    guesses = []
    for split, tau_ratio in ((0.3, 30), (0.5, 100), (0.15, 10)):
        r1, r2 = split * rp, (1 - split) * rp
        tau1 = tau2 / tau_ratio
        guesses.append([1e-8, r0, r1, tau1 ** 0.9 / r1, 0.9, r2, tau2 ** 0.85 / r2, 0.85, aw0])
    return [np.clip(g, _LB, _UB) for g in guesses]


class FitResult:
    def __init__(self, params, spec, cost):
        self.params = params
        self.spec = spec
        zm = impedance(params, spec.omega)
        rel = (zm - spec.z) / np.abs(spec.z)
        self.rel_residuals = rel
        self.rms_error = float(np.sqrt(np.mean(np.abs(rel) ** 2) / 2))
        self.cost = cost

    def value(self, name):
        if name == "Rp":
            return self.params[2] + self.params[5]
        if name == "Rtot":
            return self.params[1] + self.params[2] + self.params[5]
        return self.params[PARAM_NAMES.index(name)]

    def as_dict(self):
        d = dict(zip(PARAM_NAMES, map(float, self.params)))
        d["Rp"] = float(self.value("Rp"))
        d["Rtot"] = float(self.value("Rtot"))
        d["rms_error"] = self.rms_error
        return d


def _solve(x0, omega, z):
    return least_squares(_residual, x0, args=(omega, z), bounds=(_XLB, _XUB),
                         x_scale="jac", max_nfev=2000)


def fit(spec, start=None):
    starts = [start] if start is not None else _initial_guesses(spec)
    best = None
    for p0 in starts:
        sol = _solve(_to_x(p0), spec.omega, spec.z)
        if best is None or sol.cost < best.cost:
            best = sol
    params = _from_x(best.x)
    # Keep R1 as the faster (high-frequency) arc so the labels stay meaningful.
    t1 = (params[2] * params[3]) ** (1 / params[4])
    t2 = (params[5] * params[6]) ** (1 / params[7])
    if t1 > t2:
        params[[2, 3, 4, 5, 6, 7]] = params[[5, 6, 7, 2, 3, 4]]
    return FitResult(params, spec, best.cost)


def monte_carlo(result, n=200, seed=0, noise_floor=0.002):
    """Refit n synthetic spectra = best fit + noise sized like the real residuals.

    Returns an array (n, len(PARAM_NAMES)) of refitted parameters, plus the
    relative noise level used.
    """
    from .io import Spectrum

    spec = result.spec
    rng = np.random.default_rng(seed)
    z_fit = impedance(result.params, spec.omega)
    sigma = max(result.rms_error, noise_floor)
    samples = []
    for _ in range(n):
        noise = sigma * np.abs(z_fit) * (rng.standard_normal(len(z_fit)) + 1j * rng.standard_normal(len(z_fit)))
        synth = Spectrum(spec.freq, z_fit + noise)
        samples.append(fit(synth, start=result.params).params)
    return np.array(samples), sigma
