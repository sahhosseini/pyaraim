import json

from pyaraim.cli import main
from pyaraim.testing.refcases import check_reference_cases, generate_reference_cases


def test_reference_cases_roundtrip(tmp_path):
    f = tmp_path / "ref.json"
    cases = generate_reference_cases(f, n_random=2, seed=1)
    assert len(cases) == 2 * 3 + 1
    res = check_reference_cases(f)
    assert all(res.values())
    data = json.loads(f.read_text())
    assert data["schema"].startswith("pyaraim.refcases")
    assert {"G", "var_int", "var_acc", "p_sat"} <= set(data["cases"][0]["inputs"])


def test_reference_check_detects_tampering(tmp_path):
    f = tmp_path / "ref.json"
    generate_reference_cases(f, n_random=1, seed=1)
    data = json.loads(f.read_text())
    data["cases"][0]["expected"]["vpl"] *= 1.01
    f.write_text(json.dumps(data))
    assert not all(check_reference_cases(f).values())


def test_cli_appendix_j(capsys):
    assert main(["appendix-j"]) == 0
    out = capsys.readouterr().out
    assert "VPL" in out and "19.10" in out


def test_cli_list_almanacs(capsys):
    assert main(["list-almanacs"]) == 0
    assert "gps_24" in capsys.readouterr().out


def test_cli_availability_small(tmp_path, capsys):
    out = tmp_path / "run"
    rc = main(["availability", "--region", "conus", "--lat-step", "20", "--lon-step", "30", "--t-end", "3600",
               "--t-step", "1800", "--hal", "556", "--out", str(out)])
    assert rc == 0
    assert (tmp_path / "run.npz").exists() and (tmp_path / "run.csv").exists()
    assert "coverage" in capsys.readouterr().out


def test_cli_montecarlo_passes(capsys):
    assert main(["montecarlo", "--skies", "1", "--trials", "5000"]) == 0
