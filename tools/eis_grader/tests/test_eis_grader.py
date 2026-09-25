import json

import numpy as np
import pytest

from eis_grader import cli, grade, io, kk, ml, model, synth


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


def _population_files(tmp_path, n=40, seed=1, labeller=None):
    labels = tmp_path / "labels.csv"
    with open(labels, "w") as fh:
        fh.write("file,soh\n" if labeller is None else "file,label\n")
        for spec, soh in synth.population(n, seed=seed):
            io.save(tmp_path / f"{spec.name}.csv", spec)
            fh.write(f"{spec.name}.csv,{soh if labeller is None else labeller(soh)}\n")
    return labels


def test_random_forest_regression_beats_single_feature(tmp_path):
    files, y, task = ml.load_labels(_population_files(tmp_path))
    assert task == "regression"
    bundle = ml.train(files, y, task, repeats=2, log=lambda *_: None)
    cv = bundle["cv"]
    assert cv["random_forest"]["MAE"][0] < cv["total_resistance_only"]["MAE"][0]
    assert cv["random_forest"]["MAE"][0] < 3.0

    spec, soh = synth.population(1, seed=99)[0]
    res = model.fit(spec)
    samples, _ = model.monte_carlo(res, n=20)
    pred = ml.predict_mc(bundle, samples, (spec.freq.min(), spec.freq.max()))
    assert abs(pred["soh"] - soh) < 3 * cv["random_forest"]["MAE"][0]


def test_random_forest_classification_and_cli(tmp_path):
    import joblib

    labels = _population_files(tmp_path, n=30, labeller=lambda s: "good" if s >= 85 else "bad")
    model_path = tmp_path / "m.joblib"
    from eis_grader import train
    assert train.main([str(labels), "--out", str(model_path), "--repeats", "2"]) == 0
    assert joblib.load(model_path)["task"] == "classification"

    new = tmp_path / "new.csv"
    io.save(new, synth.spectrum(synth.aged(1.0, 2.5), seed=7))
    out = tmp_path / "out"
    assert cli.main([str(new), "--model", str(model_path), "--mc", "20", "--out", str(out)]) == 0
    rf = json.loads((out / "report.json").read_text())["cells"][0]["random_forest"]
    assert rf["label"] == "bad"
