from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
UNSUPPORTED_GITHUB_MATH = (r"\operatorname",)


def markdown_files() -> list[Path]:
    return [path for path in ROOT.rglob("*.md") if ".git" not in path.parts]


def test_docs_directory_contains_one_technical_report() -> None:
    reports = sorted(path.relative_to(ROOT) for path in (ROOT / "docs").glob("*.md"))
    assert reports == [Path("docs/technical-report.md")]


def test_markdown_blocks_are_balanced() -> None:
    for path in markdown_files():
        in_fence = False
        in_display_math = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if "$$" in line:
                assert stripped == "$$", (
                    f"{path}:{line_number}: display delimiters must be on their own line"
                )
                in_display_math = not in_display_math
        assert not in_fence, f"{path}: unclosed code fence"
        assert not in_display_math, f"{path}: unclosed display-math block"


def test_markdown_avoids_unsupported_github_math_macros() -> None:
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for macro in UNSUPPORTED_GITHUB_MATH:
            assert macro not in text, f"{path}: GitHub rejects math macro {macro}"


def test_local_markdown_links_exist() -> None:
    for path in markdown_files():
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (path.parent / target).exists(), f"{path}: missing local link {target}"


def test_public_report_has_no_private_evidence_links_or_machine_identity() -> None:
    report = (ROOT / "docs/technical-report.md").read_text(encoding="utf-8")
    forbidden = (
        "SHA-256",
        "codex/",
        "../",
        "/home/",
        "local_surface_evaluator",
        "qh_adam_trajectory_dataset_pilot_acceptance",
    )
    for value in forbidden:
        assert value not in report
