"""Synthetic battery spectra for demos and tests."""
import numpy as np

from .io import Spectrum
from .model import impedance

# Roughly an 18650 NMC cell at 50% SOC, 25 °C (ohms).
FRESH = {"L": 2e-7, "R0": 0.020, "R1": 0.006, "Q1": 0.5, "a1": 0.9,
         "R2": 0.012, "Q2": 3.0, "a2": 0.85, "Aw": 0.004}


def aged(r0_growth=1.0, rp_growth=1.0):
    p = dict(FRESH)
    p["R0"] *= r0_growth
    p["R1"] *= rp_growth
    p["R2"] *= rp_growth
    return p


def spectrum(params, noise=0.003, fmin=0.05, fmax=10e3, per_decade=10, seed=0, name="synthetic"):
    decades = np.log10(fmax / fmin)
    freq = np.logspace(np.log10(fmax), np.log10(fmin), int(decades * per_decade) + 1)
    p = [params[k] for k in ("L", "R0", "R1", "Q1", "a1", "R2", "Q2", "a2", "Aw")]
    z = impedance(p, 2 * np.pi * freq)
    rng = np.random.default_rng(seed)
    z = z + noise * np.abs(z) * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))
    return Spectrum(freq, z, name=name)


def population(n=40, seed=0):
    """Labelled cells for testing the ML path.

    SOH mainly drives charge-transfer growth and arc depression, while R0 also
    carries aging-unrelated spread (contacts, manufacturing), so total resistance
    alone is an imperfect SOH proxy.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(n):
        soh = rng.uniform(70, 100)
        fade = (100 - soh) / 30  # 0 = new, 1 = 70% SOH
        p = dict(FRESH)
        p["R0"] *= (1 + 0.3 * fade) * rng.lognormal(0, 0.15)
        p["R1"] *= (1 + 0.8 * fade) * rng.lognormal(0, 0.08)
        p["R2"] *= (1 + 1.5 * fade ** 1.5) * rng.lognormal(0, 0.08)
        p["a2"] = FRESH["a2"] - 0.1 * fade + rng.normal(0, 0.01)
        p["Aw"] *= (1 + 1.0 * fade) * rng.lognormal(0, 0.1)
        cells.append((spectrum(p, seed=seed * 1000 + i, name=f"cell{i:02d}"), soh))
    return cells
