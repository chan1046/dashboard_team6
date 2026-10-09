"""Recover a deleted source file from this task's archived FileChange events."""

from __future__ import annotations

import json
import re
from pathlib import Path

SESSION = Path(r"C:\Users\USER\.codex\sessions\2026\09\22\rollout-2026-09-22T17-41-36-01a0c847-133e-7a52-8217-d512e5355fe0.jsonl")
DEST = Path(__file__).resolve().parents[1] / "experiments" / "historical_exact_replay"
TARGET = r"LS_team6\pipeline.py"


def apply_unified(source: str, diff: str) -> str:
    lines = source.splitlines(keepends=True)
    patch_lines = diff.splitlines(keepends=True)
    index = 0
    offset = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        if not line.startswith("@@ "):
            index += 1
            continue
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if not match:
            raise ValueError(f"Unsupported hunk: {line}")
        position = int(match.group(1)) - 1 + offset
        index += 1
        before, after = [], []
        while index < len(patch_lines) and not patch_lines[index].startswith("@@ "):
            body = patch_lines[index]
            marker = body[:1]
            if marker == " ":
                before.append(body[1:]); after.append(body[1:])
            elif marker == "-":
                before.append(body[1:])
            elif marker == "+":
                after.append(body[1:])
            elif marker == "\\":
                pass
            else:
                raise ValueError(f"Unsupported diff marker {marker!r}")
            index += 1
        if lines[position:position + len(before)] != before:
            # FileChange hunks can omit the final newline or give a stale
            # numeric offset. Search for the exact context before rejecting.
            matches = [i for i in range(len(lines) - len(before) + 1) if lines[i:i + len(before)] == before]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely apply hunk at {position}; matches={matches[:5]}")
            position = matches[0]
        lines[position:position + len(before)] = after
        offset += len(after) - len(before)
    return "".join(lines)


def main() -> None:
    text = None
    updates = []
    DEST.mkdir(parents=True, exist_ok=True)
    for line_number, line in enumerate(SESSION.open(encoding="utf-8"), 1):
        try:
            item = json.loads(line).get("payload", {}).get("item", {})
        except json.JSONDecodeError:
            continue
        for filename, change in item.get("changes", {}).items():
            if not filename.endswith(TARGET):
                continue
            if change["type"] == "add":
                text = change["content"]
            elif change["type"] == "update":
                if text is None:
                    raise ValueError("Update appeared before original source")
                text = apply_unified(text, change["unified_diff"])
            updates.append((line_number, change["type"], len(text)))
            if line_number == 4983:
                (DEST / "pipeline_continuous_3_6_12_24.py").write_text(text, encoding="utf-8")
    if text is None:
        raise FileNotFoundError(TARGET)
    (DEST / "pipeline_last_recorded.py").write_text(text, encoding="utf-8")
    (DEST / "recovery_manifest.json").write_text(json.dumps({"source": str(SESSION), "updates": updates}, indent=2), encoding="utf-8")
    print(f"Recovered {len(updates)} revisions: {updates[-5:]}")


if __name__ == "__main__":
    main()
