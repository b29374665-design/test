"""Load EIS spectra from CSV / TXT exports (Gamry, BioLogic, ZView-style, plain columns)."""
import re

import numpy as np


class Spectrum:
    def __init__(self, freq, z, name=""):
        order = np.argsort(freq)[::-1]  # high -> low frequency
        self.freq = np.asarray(freq, float)[order]
        self.z = np.asarray(z, complex)[order]
        self.name = name

    @property
    def omega(self):
        return 2 * np.pi * self.freq


_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _split(line):
    for sep in ("\t", ",", ";"):
        if sep in line:
            return [c.strip().strip('"') for c in line.split(sep)]
    return line.split()


def _is_numeric_row(cells):
    return len(cells) >= 3 and all(_NUM.match(c) for c in cells if c != "")


def _pick_columns(header):
    """Return (freq, real, imag) column indices from header names, or None."""
    h = [c.lower().replace(" ", "") for c in header]
    freq = next((i for i, c in enumerate(h) if "freq" in c or c in ("f", "f/hz", "hz")), None)
    real = next((i for i, c in enumerate(h)
                 if re.search(r"re\(z\)|zreal|z'(?!')|^zre|^re\b|^real|z_re|zr\b", c)), None)
    imag = next((i for i, c in enumerate(h)
                 if re.search(r"im\(z\)|zimag|z''|^-?zim|^-?im\b|^-?imag|z_im|zi\b", c)), None)
    if None in (freq, real, imag):
        return None
    return freq, real, imag


def _unit_scale(name):
    name = name.lower()
    if "mohm" in name or "mΩ" in name or "mω" in name:
        return 1e-3
    return 1.0


def load(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n\r") for ln in fh if ln.strip()]

    header, rows = None, []
    for ln in lines:
        cells = _split(ln)
        if _is_numeric_row(cells):
            rows.append([float(c) if c else np.nan for c in cells])
        elif not rows:
            header = cells  # last non-numeric line before data is the header

    if not rows:
        raise ValueError(f"{path}: 找不到數值資料")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows])

    cols = _pick_columns(header) if header else None
    if cols is None or max(cols) >= width:
        cols, header = (0, 1, 2), None  # assume freq, Z', Z''
    fi, ri, ii = cols
    scale = _unit_scale(header[ri]) if header else 1.0

    freq = data[:, fi]
    zr = data[:, ri] * scale
    zi = data[:, ii] * scale
    ok = np.isfinite(freq) & np.isfinite(zr) & np.isfinite(zi) & (freq > 0)
    freq, zr, zi = freq[ok], zr[ok], zi[ok]

    # Battery arcs are capacitive (Im Z < 0) over most of the spectrum. Exports
    # often store -Im(Z); normalise to the physical sign from the data itself.
    if np.median(zi) > 0:
        zi = -zi
    return Spectrum(freq, zr + 1j * zi, name=str(path))


def save(path, spec):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("freq_Hz,Zreal_Ohm,-Zimag_Ohm\n")
        for f, z in zip(spec.freq, spec.z):
            fh.write(f"{f:.6g},{z.real:.8g},{-z.imag:.8g}\n")
