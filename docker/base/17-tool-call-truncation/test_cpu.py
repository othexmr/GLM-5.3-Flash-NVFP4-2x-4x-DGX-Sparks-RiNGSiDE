# SPDX-License-Identifier: Apache-2.0
"""SL-01 gate: a tool call truncated inside <arg_value> must still deliver
`arguments` that parse as JSON, must deliver the same text on the streaming
and the non-streaming path, and must be reported as a truncation.

CPU only, no GPU, no service, no network.

    python3 test_cpu.py                 # runs both variants, prints a summary
    python3 test_cpu.py --variant fixed # one variant (used by the runner)
    python3 test_cpu.py --json          # machine-readable

Inside the tested image the same file runs against the installed vLLM
(``import vllm.parser.glm53_moe`` succeeds, so ``--variant`` is ignored and
what is installed is what is graded).  Threshold: 0 failures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import Request, Tokenizer, Tool, load_parser_cls  # noqa: E402

THINK = "Working.</think>"

SCHEMA_PATH = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "mode": {"type": "string"}},
}
SCHEMA_COUNT = {"type": "object", "properties": {"count": {"type": "integer"}}}

TOOLS_PATH = [Tool("inspect", SCHEMA_PATH)]
TOOLS_COUNT = [Tool("inspect", SCHEMA_COUNT)]

# (label, model output, tools, expects_truncation)
CASES = [
    (
        "trunc-in-arg-value",
        THINK + "<tool_call>inspect<arg_key>path</arg_key><arg_value>/etc/passwd",
        TOOLS_PATH,
        True,
    ),
    (
        "trunc-in-arg-value-no-schema",
        THINK + "<tool_call>inspect<arg_key>path</arg_key><arg_value>/etc/passwd",
        None,
        True,
    ),
    (
        "trunc-in-second-arg-value",
        THINK
        + "<tool_call>inspect<arg_key>path</arg_key><arg_value>/a</arg_value>"
        + "<arg_key>mode</arg_key><arg_value>re",
        TOOLS_PATH,
        True,
    ),
    (
        "trunc-in-integer-arg-value",
        THINK + "<tool_call>inspect<arg_key>count</arg_key><arg_value>12",
        TOOLS_COUNT,
        True,
    ),
    (
        "trunc-with-quote-in-value",
        THINK + '<tool_call>inspect<arg_key>path</arg_key><arg_value>a"b\\c',
        TOOLS_PATH,
        True,
    ),
    (
        "trunc-in-arg-key",
        THINK + "<tool_call>inspect<arg_key>pa",
        TOOLS_PATH,
        True,
    ),
    (
        "trunc-after-tool-name",
        THINK + "<tool_call>inspect",
        TOOLS_PATH,
        True,
    ),
    # Malformed-but-complete body: the model repeated one <arg_key>, so the
    # later value cannot extend the prefix already streamed.  Nothing can
    # recover the second value once the first is on the wire; the gate only
    # requires that what is delivered is valid JSON and that both paths say
    # the same thing.  Flagged as truncated because the parser, not the
    # model, closed the arguments.
    (
        "repeated-arg-key",
        THINK
        + "<tool_call>inspect<arg_key>path</arg_key><arg_value>/a</arg_value>"
        + "<arg_key>path</arg_key><arg_value>/b</arg_value></tool_call>",
        TOOLS_PATH,
        True,
    ),
    # Regression guards: untruncated calls must be unchanged.
    (
        "complete-single-arg",
        THINK
        + "<tool_call>inspect<arg_key>path</arg_key>"
        + "<arg_value>/etc/passwd</arg_value></tool_call>",
        TOOLS_PATH,
        False,
    ),
    (
        "complete-two-args",
        THINK
        + "<tool_call>inspect<arg_key>path</arg_key><arg_value>/a</arg_value>"
        + "<arg_key>mode</arg_key><arg_value>re</arg_value></tool_call>",
        TOOLS_PATH,
        False,
    ),
    (
        "complete-integer-arg",
        THINK
        + "<tool_call>inspect<arg_key>count</arg_key>"
        + "<arg_value>12</arg_value></tool_call>",
        TOOLS_COUNT,
        False,
    ),
    (
        "complete-no-args",
        THINK + "<tool_call>inspect</tool_call>",
        TOOLS_PATH,
        False,
    ),
]


def _stream(parser_cls, tok, text, tools, chunks, use_ids):
    """Feed *chunks* and return (name, arguments, truncated_flag_or_None)."""
    request = Request(tools=tools)
    parser = parser_cls(tok, tools=tools, chat_template_kwargs={})
    name_parts, arg_parts = [], []
    last = len(chunks) - 1
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            chunk,
            tok.encode(chunk) if use_ids else [],
            request,
            prompt_token_ids=tok.encode("<think>"),
            finished=i == last,
        )
        if delta is None:
            continue
        for tc in delta.tool_calls or []:
            if tc.function is None:
                continue
            if tc.function.name:
                name_parts.append(tc.function.name)
            if tc.function.arguments:
                arg_parts.append(tc.function.arguments)
    truncated = getattr(parser, "tool_call_truncated", None)
    return "".join(name_parts), "".join(arg_parts), truncated


def _nonstream(parser_cls, tok, text, tools):
    request = Request(tools=tools)
    parser = parser_cls(tok, tools=tools, chat_template_kwargs={})
    _, _, calls = parser.parse(text, request, enable_auto_tools=True)
    if not calls:
        return None, None, getattr(parser, "tool_call_truncated", None)
    return calls[0].name, calls[0].arguments, getattr(
        parser, "tool_call_truncated", None
    )


def _splits(text, tok, use_ids):
    """Chunkings to exercise: whole, per-chunk boundaries, per-token."""
    if not use_ids:
        yield [text]
        for i in range(1, len(text)):
            yield [text[:i], text[i:]]
        yield list(text)
        return
    # Token-boundary splits only: a real detokenizer never hands half a
    # special token with that token's id.
    ids = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]
    yield [text]
    for i in range(1, len(pieces)):
        yield ["".join(pieces[:i]), "".join(pieces[i:])]
    yield pieces


def run(variant):
    parser_cls, mode = load_parser_cls(variant)
    tok = Tokenizer()
    failures = []
    checked = 0

    for label, text, tools, expect_trunc in CASES:
        ns_name, ns_args, _ = _nonstream(parser_cls, tok, text, tools)
        for use_ids in (False, True):
            for chunks in _splits(text, tok, use_ids):
                checked += 1
                name, args, truncated = _stream(
                    parser_cls, tok, text, tools, chunks, use_ids
                )
                where = f"{label}|ids={use_ids}|n={len(chunks)}"

                if name != "inspect":
                    failures.append(
                        {"case": where, "check": "tool name", "got": name}
                    )
                    continue
                try:
                    json.loads(args)
                except (ValueError, TypeError) as exc:
                    failures.append(
                        {
                            "case": where,
                            "check": "streamed arguments parse as JSON",
                            "got": args,
                            "error": str(exc),
                        }
                    )
                if args != ns_args:
                    failures.append(
                        {
                            "case": where,
                            "check": "streaming == non-streaming arguments",
                            "got": args,
                            "non_streaming": ns_args,
                        }
                    )
                if truncated is None:
                    failures.append(
                        {
                            "case": where,
                            "check": "parser exposes tool_call_truncated",
                            "got": None,
                        }
                    )
                elif bool(truncated) != expect_trunc:
                    failures.append(
                        {
                            "case": where,
                            "check": "truncation reported",
                            "got": bool(truncated),
                            "expected": expect_trunc,
                        }
                    )
        if ns_name != "inspect":
            failures.append(
                {"case": label + "|non-streaming", "check": "tool name", "got": ns_name}
            )

    by_check: dict[str, int] = {}
    for f in failures:
        by_check[f["check"]] = by_check.get(f["check"], 0) + 1

    return {
        "variant": variant,
        "mode": mode,
        "checks": checked,
        "failures": len(failures),
        "failures_by_check": by_check,
        "first_failures": failures[:4],
        "pass": not failures,
    }


def dump(variant):
    """Per-case streamed vs non-streamed values, single-chunk feed."""
    parser_cls, mode = load_parser_cls(variant)
    tok = Tokenizer()
    rows = []
    for label, text, tools, expect_trunc in CASES:
        name, args, truncated = _stream(parser_cls, tok, text, tools, [text], False)
        _, ns_args, _ = _nonstream(parser_cls, tok, text, tools)
        try:
            json.loads(args)
            valid = True
        except (ValueError, TypeError):
            valid = False
        rows.append(
            {
                "case": label,
                "streamed_arguments": args,
                "valid_json": valid,
                "non_streaming_arguments": ns_args,
                "agree": args == ns_args,
                "truncated_flag": truncated,
                "expected_truncated": expect_trunc,
            }
        )
    return {"variant": variant, "mode": mode, "rows": rows}


def emit(variant):
    """Every streamed (name, arguments) for the untruncated cases.

    Diffing this between the two variants is the regression evidence that
    the fix changes nothing on a tool call the model actually closed.
    """
    parser_cls, _ = load_parser_cls(variant)
    tok = Tokenizer()
    out = []
    for label, text, tools, expect_trunc in CASES:
        if expect_trunc:
            continue
        for use_ids in (False, True):
            for n, chunks in enumerate(_splits(text, tok, use_ids)):
                name, args, _ = _stream(parser_cls, tok, text, tools, chunks, use_ids)
                out.append([label, use_ids, n, name, args])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("original", "fixed"))
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    if args.dump:
        print(json.dumps(dump(args.variant or "original"), indent=2))
        return 0
    if args.emit:
        print(json.dumps(emit(args.variant or "original")))
        return 0

    if args.variant:
        result = run(args.variant)
        print(json.dumps(result, indent=2))
        return 0 if result["pass"] else 1

    results = []
    for variant in ("original", "fixed"):
        proc = subprocess.run(
            [sys.executable, __file__, "--variant", variant],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        try:
            results.append(json.loads(proc.stdout))
        except ValueError:
            results.append({"variant": variant, "pass": False, "failures": -1})

    emitted = {}
    for variant in ("original", "fixed"):
        proc = subprocess.run(
            [sys.executable, __file__, "--emit", "--variant", variant],
            capture_output=True,
            text=True,
        )
        emitted[variant] = json.loads(proc.stdout) if proc.stdout else None

    original, fixed = results
    verdict = {
        "gate": "SL-01",
        "original_fails": not original["pass"],
        "fixed_passes": fixed["pass"],
        "original_failures": original["failures"],
        "original_failures_by_check": original.get("failures_by_check"),
        "fixed_failures": fixed["failures"],
        "checks_per_variant": original.get("checks"),
        "untruncated_streams_compared": len(emitted["original"] or []),
        "untruncated_streams_byte_identical": emitted["original"] == emitted["fixed"],
        "pass": (not original["pass"])
        and fixed["pass"]
        and emitted["original"] == emitted["fixed"],
    }
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
