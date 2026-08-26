from __future__ import annotations

from flow_matching.optimization import QH_OPTIMIZATION_DEFAULTS
from scripts.optimize_flow_latent import build_parser as optimizer_parser
from scripts.screen_flow_starts import build_parser as screening_parser


def test_validated_309_trajectory_protocol_is_the_public_default() -> None:
    defaults = QH_OPTIMIZATION_DEFAULTS
    assert defaults.candidate_count == 32
    assert defaults.iterations == 200
    assert defaults.directions == 64
    assert defaults.perturbation == 0.005
    assert defaults.learning_rate == 0.02
    assert defaults.beta1 == 0.7
    assert defaults.beta2 == 0.999
    assert defaults.flow_steps == 128
    assert defaults.gradient_mode == "random-orthogonal"


def test_command_line_defaults_use_the_validated_protocol() -> None:
    screening = screening_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--lib",
            "libstellarator_gpu.so",
            "--out-dir",
            "screen",
            "--nfp",
            "4",
            "--n-base-coils",
            "3",
            "--seed",
            "1",
        ]
    )
    optimization = optimizer_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--initial-case",
            "screen/selected_start.json",
            "--lib",
            "libstellarator_gpu.so",
            "--out-dir",
            "optimized",
        ]
    )

    assert screening.candidate_count == QH_OPTIMIZATION_DEFAULTS.candidate_count
    assert screening.flow_steps == QH_OPTIMIZATION_DEFAULTS.flow_steps
    assert optimization.iterations == QH_OPTIMIZATION_DEFAULTS.iterations
    assert optimization.random_directions == QH_OPTIMIZATION_DEFAULTS.directions
    assert optimization.gradient_mode == "random-orthogonal"
    assert optimization.optimizer == "adam"
    assert optimization.perturbation == QH_OPTIMIZATION_DEFAULTS.perturbation
    assert optimization.learning_rate == QH_OPTIMIZATION_DEFAULTS.learning_rate
    assert optimization.beta1 == QH_OPTIMIZATION_DEFAULTS.beta1
    assert optimization.beta2 == QH_OPTIMIZATION_DEFAULTS.beta2
    assert optimization.flow_steps == QH_OPTIMIZATION_DEFAULTS.flow_steps
