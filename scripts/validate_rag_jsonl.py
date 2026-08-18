#!/usr/bin/env python3
"""Structural checks for a RAG JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyserini_client import load_topics_tsv, project_root

ALLOWED_META = frozenset(
    {"team_id", "narrative_id", "narrative", "run_id", "run_desc"}
)
WORD_LIMIT = 1024
MAX_CITES = 3


def validate_record(
    obj: dict,
    *,
    line_no: int,
    expected_narrative: str | None,
) -> list[str]:
    errs: list[str] = []
    prefix = f"line {line_no}"

    if not isinstance(obj, dict):
        return [f"{prefix}: not a JSON object"]

    for key in ("metadata", "references", "answer"):
        if key not in obj:
            errs.append(f"{prefix}: missing '{key}'")

    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        errs.append(f"{prefix}: metadata must be an object")
        return errs

    extra = set(meta) - ALLOWED_META
    missing = ALLOWED_META - set(meta)
    if extra:
        errs.append(f"{prefix}: extra metadata keys {sorted(extra)}")
    if missing:
        errs.append(f"{prefix}: missing metadata keys {sorted(missing)}")

    for k in ("team_id", "narrative_id", "narrative", "run_id", "run_desc"):
        if k in meta and not isinstance(meta[k], str):
            errs.append(f"{prefix}: metadata.{k} must be a string")

    if expected_narrative is not None and meta.get("narrative") != expected_narrative:
        errs.append(f"{prefix}: narrative text does not match topics file")

    refs = obj.get("references")
    if not isinstance(refs, list):
        errs.append(f"{prefix}: references must be a list")
        refs = []
    else:
        for i, r in enumerate(refs):
            if not isinstance(r, str) or not r.strip():
                errs.append(f"{prefix}: references[{i}] must be a non-empty string")

    answer = obj.get("answer")
    if not isinstance(answer, list):
        errs.append(f"{prefix}: answer must be a list")
        return errs
    if not answer:
        errs.append(f"{prefix}: answer is empty")

    words = 0
    cited: set[int] = set()
    for si, sent in enumerate(answer):
        if not isinstance(sent, dict):
            errs.append(f"{prefix}: answer[{si}] must be an object")
            continue
        text = sent.get("text")
        cites = sent.get("citations")
        if not isinstance(text, str) or not text.strip():
            errs.append(f"{prefix}: answer[{si}].text missing/empty")
        else:
            words += len(text.split())
        if not isinstance(cites, list):
            errs.append(f"{prefix}: answer[{si}].citations must be a list")
            continue
        if len(cites) > MAX_CITES:
            errs.append(
                f"{prefix}: answer[{si}] has {len(cites)} citations (max {MAX_CITES})"
            )
        for c in cites:
            if not isinstance(c, int) or isinstance(c, bool):
                errs.append(f"{prefix}: answer[{si}] citation {c!r} not an int")
                continue
            if c < 0 or c >= len(refs):
                errs.append(
                    f"{prefix}: answer[{si}] citation {c} out of range "
                    f"(refs={len(refs)})"
                )
            else:
                cited.add(c)

    if words > WORD_LIMIT:
        errs.append(f"{prefix}: {words} words exceeds {WORD_LIMIT}")

    unused = [i for i in range(len(refs)) if i not in cited]
    if unused:
        errs.append(f"{prefix}: uncited references indices {unused}")

    return errs


def main() -> int:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument(
        "--topics",
        type=Path,
        default=None,
        help="Optional topics TSV to enforce narrative_id coverage and exact narrative text.",
    )
    args = ap.parse_args()

    topics_map: dict[str, str] | None = None
    if args.topics is not None:
        topics_map = {tid: text for tid, text in load_topics_tsv(args.topics)}

    errors: list[str] = []
    seen: set[str] = set()
    n = 0
    for line_no, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue
        n += 1
        tid = None
        if isinstance(obj, dict) and isinstance(obj.get("metadata"), dict):
            tid = str(obj["metadata"].get("narrative_id", ""))
            if tid:
                if tid in seen:
                    errors.append(f"line {line_no}: duplicate narrative_id {tid}")
                seen.add(tid)
        expected = None
        if topics_map is not None and tid:
            if tid not in topics_map:
                errors.append(f"line {line_no}: unknown narrative_id {tid}")
            else:
                expected = topics_map[tid]
        errors.extend(validate_record(obj, line_no=line_no, expected_narrative=expected))

    if topics_map is not None:
        missing_topics = sorted(set(topics_map) - seen)
        if missing_topics:
            errors.append(f"missing topics in output: {missing_topics}")

    print(f"Validated {n} records from {args.input}")
    if errors:
        print(f"FAIL: {len(errors)} issue(s)")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
