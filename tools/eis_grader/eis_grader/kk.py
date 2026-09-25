"""Lin-KK data-validity test (Schönleber et al., Electrochim. Acta 131 (2014) 20).

A spectrum that cannot be described by a series of RC elements violates
Kramers-Kronig (drift, non-linearity, non-steady state), and any grade derived
from it is unreliable.
"""
import numpy as np


def _design(omega, taus):
    jw = 1j * omega
    cols = [np.ones_like(jw), jw, 1 / jw]  # R_ohm, L, series C (diffusion tail)
    cols += [1 / (1 + jw * t) for t in taus]
    return np.column_stack(cols)


def lin_kk(spec, mu_threshold=0.85, max_m=None):
    omega, z = spec.omega, spec.z
    w = 1 / np.abs(z)
    max_m = max_m or max(3, len(omega) // 2)
    for m in range(3, max_m + 1):
        taus = np.logspace(np.log10(1 / omega.max()), np.log10(1 / omega.min()), m)
        a = _design(omega, taus)
        aw = a * w[:, None]
        lhs = np.vstack([aw.real, aw.imag])
        rhs = np.concatenate([(z * w).real, (z * w).imag])
        coef, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        rk = coef[3:]
        pos, neg = rk[rk > 0].sum(), -rk[rk < 0].sum()
        mu = 1 - neg / pos if pos > 0 else 0
        if mu <= mu_threshold:
            break
    zk = a @ coef
    rel = (z - zk) / np.abs(z)
    return {
        "M": m,
        "mu": float(mu),
        "max_residual": float(max(np.abs(rel.real).max(), np.abs(rel.imag).max())),
        "rms_residual": float(np.sqrt(np.mean(np.abs(rel) ** 2) / 2)),
        "residuals": rel,
    }
