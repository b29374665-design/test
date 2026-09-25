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
