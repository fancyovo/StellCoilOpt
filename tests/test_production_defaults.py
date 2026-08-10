from __future__ import annotations

from pathlib import Path

from stellarator_eval.config import VolumeQSConfig


ROOT = Path(__file__).resolve().parents[1]


def test_python_volume_qs_uses_cubic_iota_by_default():
    assert VolumeQSConfig().iota_degree == 3


def test_native_score_source_defaults_match_production_contract():
    source = (ROOT / "gpu_backend/src/score_pipeline.cu").read_text(encoding="utf-8")
    for name in ("psi_n_r", "psi_n_z", "psi_n_phi"):
        assert f"config->{name} = 48;" in source
    assert "config->iota_degree = 3;" in source
    assert "axis_hint_require_continuation > 2" in source


def test_optimizer_uses_strict_mixed_axis_continuation():
    source = (ROOT / "scripts/optimize_flow_latent.py").read_text(encoding="utf-8")
    assert '"axis_hint_require_continuation": 2' in source
