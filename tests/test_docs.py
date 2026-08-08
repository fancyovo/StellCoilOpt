from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISALLOWED_MATH = (r"\operatorname", r"\text", r"\qquad")
TRAILING_INLINE_MATH = set("，。；：、！？,.!?:;)]}|")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_prose_lines():
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue

        in_fence = False
        in_display_math = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            if "$$" in line:
                assert stripped == "$$", f"{path}:{line_number}: put display delimiters on their own lines"
                in_display_math = not in_display_math
                continue
            yield path, line_number, line, in_display_math

        assert not in_fence, f"{path}: unclosed code fence"
        assert not in_display_math, f"{path}: unclosed display-math block"


def test_markdown_math_uses_supported_multiline_syntax():
    for path, line_number, line, _ in markdown_prose_lines():
        for macro in DISALLOWED_MATH:
            assert macro not in line, f"{path}:{line_number}: unsupported or unnecessary macro {macro}"


def test_inline_math_is_delimited_and_separated_from_prose():
    errors = []
    for path, line_number, line, in_display_math in markdown_prose_lines():
        if in_display_math:
            continue

        delimiters = [
            index
            for index, char in enumerate(line)
            if char == "$" and (index == 0 or line[index - 1] != "\\")
        ]
        if len(delimiters) % 2:
            errors.append(f"{path}:{line_number}: unpaired inline-math delimiter")
            continue

        for opening, closing in zip(delimiters[::2], delimiters[1::2]):
            if opening and not line[opening - 1].isspace():
                errors.append(f"{path}:{line_number}: add a space before inline math")
            if closing + 1 < len(line):
                following = line[closing + 1]
                if not (following.isspace() or following in TRAILING_INLINE_MATH):
                    errors.append(f"{path}:{line_number}: add a space after inline math")

    assert not errors, "\n".join(errors)


def test_local_markdown_links_exist():
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (path.parent / target).exists(), f"{path}: missing local link {target}"
