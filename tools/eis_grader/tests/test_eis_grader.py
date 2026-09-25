import json

import numpy as np
import pytest

from eis_grader import cli, grade, io, kk, model, synth


def _write(path, header, spec, neg_imag=True, scale=1.0, sep=","):
    with open(path, "w") as fh:
        if header:
            fh.write(header + "\n")
        for f, z in zip(spec.freq, spec.z):
            im = -z.imag if neg_imag else z.imag
            fh.write(sep.join(f"{v:.8g}" for v in (f, z.real * scale, im * scale)) + "\n")


@pytest.fixture
def fresh():
    return synth.spectrum(synth.FRESH, seed=1)


@pytest.mark.parametrize("header,neg,scale,sep", [
    ("freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm", True, 1.0, "\t"),      # BioLogic
    ("Freq,Zreal,Zimag", False, 1.0, ","),                    # Gamry
    ("", True, 1.0, " "),                                     # bare columns
    ("Frequency (Hz);Z' (mOhm);-Z'' (mOhm)", True, 1e3, ";"),  # milliohm export
])
def test_load_formats(tmp_path, fresh, header, neg, scale, sep):
    p = tmp_path / "cell.txt"
    _write(p, header, fresh, neg, scale, sep)
    spec = io.load(p)
    assert np.allclose(spec.z, fresh.z, rtol=1e-6)
    assert np.allclose(spec.freq, fresh.freq)


def test_fit_recovers_parameters(fresh):
    res = model.fit(fresh)
    assert res.value("R0") == pytest.approx(synth.FRESH["R0"], rel=0.02)
    assert res.value("Rp") == pytest.approx(synth.FRESH["R1"] + synth.FRESH["R2"], rel=0.05)
    assert res.rms_error < 0.01


def test_kk_flags_drifting_measurement(fresh):
    assert kk.lin_kk(fresh)["max_residual"] < cli.KK_LIMIT
    drift = io.Spectrum(fresh.freq, fresh.z + np.linspace(0, 0.004, len(fresh.z)))
    assert kk.lin_kk(drift)["max_residual"] > cli.KK_LIMIT


@pytest.mark.parametrize("growth,expected", [(1.1, "A"), (1.4, "B"), (1.8, "C"), (2.5, "D")])
def test_grades(growth, expected):
    spec = synth.spectrum(synth.aged(growth, growth), seed=3)
    a = cli.analyse(spec, n_mc=30, seed=0)
    res = a["fit"]
    base = {"R0": synth.FRESH["R0"], "Rp": synth.FRESH["R1"] + synth.FRESH["R2"]}
    v = grade.evaluate({"R0": res.value("R0"), "Rp": res.value("Rp")}, a["mc"], base)
    assert v["grade"] == expected


def test_borderline_cell_splits_probability():
    # Rp grown to exactly the A/B boundary with noisy data: MC should hedge.
    spec = synth.spectrum(synth.aged(1.0, 1.25), noise=0.02, seed=5)
    a = cli.analyse(spec, n_mc=100, seed=0)
    base = {"R0": synth.FRESH["R0"], "Rp": synth.FRESH["R1"] + synth.FRESH["R2"]}
    res = a["fit"]
    v = grade.evaluate({"R0": res.value("R0"), "Rp": res.value("Rp")}, a["mc"], base)
    assert 0.1 < v["probabilities"]["A"] < 0.9
    assert 0.1 < v["probabilities"]["B"] < 0.9


def test_soh():
    assert grade.soh_r(1.0, 1.0) == 100
    assert grade.soh_r(1.5, 1.0) == 50
    assert grade.soh_r(3.0, 1.0) == 0


def test_cli_batch_mode(tmp_path):
    files = []
    for i, g in enumerate((1.0, 1.02, 0.98, 2.4)):
        p = tmp_path / f"cell{i}.csv"
        io.save(p, synth.spectrum(synth.aged(g, g), seed=i))
        files.append(str(p))
    out = tmp_path / "out"
    assert cli.main(files + ["--mc", "20", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    grades = [c["verdict"]["grade"] for c in rep["cells"]]
    assert grades[:3] == ["A", "A", "A"] and grades[3] == "D"
    assert (out / "cell3.png").exists()
