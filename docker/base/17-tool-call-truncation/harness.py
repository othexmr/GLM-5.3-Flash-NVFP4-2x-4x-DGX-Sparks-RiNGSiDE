# SPDX-License-Identifier: Apache-2.0
"""Load the real GLM-5.3 parser engine on a machine with no vLLM install.

Two modes:

* ``real``  — ``import vllm.parser.glm53_moe`` from the installed package.
  Used inside the tested image, where the whole serving stack is present.
  This is the authoritative mode; ``variant`` is ignored because what is
  installed is what is tested.
* ``shim``  — used on a Mac with stdlib + nothing else.  The *engine*
  modules are the real staged files, loaded by path:

      vllm/parser/engine/events.py
      vllm/parser/engine/parser_engine_config.py
      vllm/parser/engine/incremental_lexer.py
      vllm/parser/engine/token_id_scanner.py
      vllm/parser/engine/streaming_parser_engine.py   <- variant-selected
      vllm/parser/engine/parser_engine.py             <- variant-selected
      vllm/parser/glm47_moe.py

  Only the leaves that would drag in torch / openai / pydantic are
  replaced by shims, and each shim is listed in ``SHIMMED`` below with
  what it stands in for.  ``vllm.parser.glm53_moe`` sets
  ``thinking=True, enable_thinking=True`` on ``Glm47MoeParser``
  (runtime-source/vllm/parser/glm53_moe.py:12-18) and adds adapters that
  do not touch argument handling, so the shim builds ``Glm47MoeParser``
  with thinking enabled and that is the same parser object the
  ``--tool-call-parser glm53`` path constructs.

The three things that decide this test's outcome — ``_glm47_arg_converter``,
``_safe_arg_prefix``/``_compute_arg_delta``, and
``_flush_arg_converter``/``finish()`` — are all real code in both modes.
"""

from __future__ import annotations

import importlib.util
import re as _stdlib_re
import sys
import types
from pathlib import Path

PACKET_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = (
    PACKET_DIR.parent.parent.parent / "runtime-source"
)  # work/nvfp4-workflows/runtime-source

SHIMMED = {
    "regex": "aliased to stdlib `re`; the engine only uses re.compile/re.escape "
    "(incremental_lexer.py:291, parser_engine.py:1014)",
    "vllm.logger": "init_logger -> a no-op debug logger",
    "vllm.entrypoints.chat_utils": "get_tool_call_id_type / make_tool_call_id",
    "vllm.entrypoints.generate.base.protocol": "DeltaToolCall / DeltaFunctionCall / "
    "DeltaMessage / ToolCall / FunctionCall / ExtractedToolCallInformation as "
    "plain classes instead of pydantic models",
    "vllm.entrypoints.openai.chat_completion.protocol": "ChatCompletionRequest name only",
    "vllm.entrypoints.openai.responses.protocol": "ResponsesRequest name only",
    "vllm.parser.abstract_parser": "Parser base + StreamState "
    "(fields copied from runtime-source/vllm/parser/abstract_parser.py:45-59)",
    "vllm.tool_parsers.utils": "extract_types_from_schema / coerce_to_schema_type / "
    "_TYPE_ALIASES / _is_json_finite extracted verbatim by AST from "
    "runtime-source/vllm/tool_parsers/utils.py; find_tool_name / "
    "find_tool_properties reimplemented for the plain tool objects this test "
    "builds (the real ones isinstance-match openai/pydantic tool models)",
}

SPECIAL = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]


class Tokenizer:
    """Synthetic tokenizer, same shape as image/12-reasoning/test_parser.py:10-21."""

    all_special_tokens = SPECIAL
    all_special_ids = list(range(1000, 1000 + len(SPECIAL)))

    def get_vocab(self):
        return {
            **{chr(i): i for i in range(256)},
            **dict(zip(self.all_special_tokens, self.all_special_ids)),
        }

    def encode(self, text, **kwargs):
        out = []
        while text:
            special = next((s for s in SPECIAL if text.startswith(s)), None)
            if special:
                out.append(self.get_vocab()[special])
                text = text[len(special) :]
            else:
                out.append(ord(text[0]))
                text = text[1:]
        return out

    def decode(self, ids, **kwargs):
        table = dict(zip(self.all_special_ids, SPECIAL))
        return "".join(table.get(i, chr(i)) for i in ids)


class Function:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters


class Tool:
    """Stands in for ChatCompletionToolsParam (.function.name/.parameters)."""

    def __init__(self, name, parameters):
        self.function = Function(name, parameters)


class Request:
    """Only the attributes the parser actually reads."""

    def __init__(self, tools=None, tool_choice="auto", include_reasoning=True):
        self.tools = tools
        self.tool_choice = tool_choice
        self.include_reasoning = include_reasoning
        self.skip_special_tokens = True
        self.stream = True


# ── shim construction ────────────────────────────────────────────────


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _extract_verbatim(path: Path, names):
    """exec the named top-level defs/assignments from *path*, nothing else."""
    import ast

    src = path.read_text()
    tree = ast.parse(src)
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            pieces.append(ast.get_source_segment(src, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if getattr(target, "id", None) in names:
                pieces.append(ast.get_source_segment(src, node))
    ns = {}
    exec("import json, math\n" + "\n\n".join(pieces), ns)  # noqa: S102
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"could not extract {missing} from {path}")
    return ns


def _install_shims(source_root: Path):
    sys.modules.setdefault("regex", _stdlib_re)

    for pkg in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.generate",
        "vllm.entrypoints.generate.base",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.responses",
        "vllm.parser",
        "vllm.parser.engine",
        "vllm.tokenizers",
        "vllm.tool_parsers",
    ):
        if pkg not in sys.modules:
            m = _mod(pkg)
            m.__path__ = []

    logger_mod = _mod("vllm.logger")

    class _Logger:
        def debug(self, *a, **k):
            pass

        warning = error = info = debug

    logger_mod.init_logger = lambda name: _Logger()

    chat_utils = _mod("vllm.entrypoints.chat_utils")
    chat_utils.get_tool_call_id_type = lambda model_config: "random"
    _counter = {"n": 0}

    def make_tool_call_id(id_type="random", func_name=None, idx=None):
        _counter["n"] += 1
        return f"call_{_counter['n']:06d}"

    chat_utils.make_tool_call_id = make_tool_call_id

    proto = _mod("vllm.entrypoints.generate.base.protocol")

    class DeltaFunctionCall:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments

    class DeltaToolCall:
        def __init__(self, index=0, id=None, type=None, function=None):
            self.index = index
            self.id = id
            self.type = type
            self.function = function

    class DeltaMessage:
        def __init__(self, content=None, reasoning=None, tool_calls=None):
            self.content = content
            self.reasoning = reasoning
            self.tool_calls = tool_calls

    class FunctionCall:
        def __init__(self, name=None, arguments=None, id=None):
            self.name = name
            self.arguments = arguments
            self.id = id

    class ToolCall:
        def __init__(self, id=None, function=None, type="function"):
            self.id = id
            self.function = function
            self.type = type

    class ExtractedToolCallInformation:
        def __init__(self, tools_called=False, tool_calls=(), content=None):
            self.tools_called = tools_called
            self.tool_calls = list(tool_calls)
            self.content = content

    for cls in (
        DeltaFunctionCall,
        DeltaToolCall,
        DeltaMessage,
        FunctionCall,
        ToolCall,
        ExtractedToolCallInformation,
    ):
        setattr(proto, cls.__name__, cls)

    cc_proto = _mod("vllm.entrypoints.openai.chat_completion.protocol")
    cc_proto.ChatCompletionRequest = Request
    r_proto = _mod("vllm.entrypoints.openai.responses.protocol")
    r_proto.ResponsesRequest = Request

    tokenizers = _mod("vllm.tokenizers")
    tokenizers.TokenizerLike = object

    abstract = _mod("vllm.parser.abstract_parser")

    class StreamState:
        # runtime-source/vllm/parser/abstract_parser.py:45-59
        def __init__(self, tool_call_id_type="random"):
            self.reasoning_ended = False
            self.tool_call_text_started = False
            self.prompt_reasoning_checked = False
            self.previous_text = ""
            self.previous_token_ids = []
            self.history_tool_call_cnt = 0
            self.history_tool_call_cnt_initialized = False
            self.tool_call_id_type = tool_call_id_type
            self.function_name_returned = False
            self.engine_based = False

    class Parser:
        def _initialize_history_tool_call_cnt(self, request):
            # runtime-source/vllm/parser/abstract_parser.py:177-188: for every
            # id type except kimi_k2 this only flips the initialised flag.
            self._stream_state.history_tool_call_cnt_initialized = True

    abstract.StreamState = StreamState
    abstract.Parser = Parser

    tp_utils = _mod("vllm.tool_parsers.utils")
    real = _extract_verbatim(
        source_root / "vllm/tool_parsers/utils.py",
        [
            "extract_types_from_schema",
            "coerce_to_schema_type",
            "_TYPE_ALIASES",
            "_is_json_finite",
        ],
    )
    tp_utils.extract_types_from_schema = real["extract_types_from_schema"]
    tp_utils.coerce_to_schema_type = real["coerce_to_schema_type"]

    def find_tool_properties(tools, tool_name):
        if not tools:
            return {}
        for tool in tools:
            fn = getattr(tool, "function", tool)
            if getattr(fn, "name", None) == tool_name:
                return (getattr(fn, "parameters", None) or {}).get("properties", {})
        return {}

    def find_tool_name(tools, tool_name):
        if not tools:
            return False
        return any(
            getattr(getattr(t, "function", t), "name", None) == tool_name for t in tools
        )

    tp_utils.find_tool_properties = find_tool_properties
    tp_utils.find_tool_name = find_tool_name

    atp = _mod("vllm.tool_parsers.abstract_tool_parser")
    atp.Tool = Tool
    atp.ToolParser = object


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_parser_cls(variant: str, source_root: Path | None = None):
    """Return (Glm53-equivalent parser class, mode string)."""
    try:
        from vllm.parser.glm53_moe import Glm53MoeParser  # noqa: PLC0415

        return Glm53MoeParser, "real"
    except Exception:  # noqa: BLE001 - no vLLM on this machine; use the shim
        pass

    source_root = Path(source_root or DEFAULT_SOURCE_ROOT)
    variant_root = PACKET_DIR / variant  # "original" or "fixed"
    if not (variant_root / "vllm/parser/engine/parser_engine.py").is_file():
        raise SystemExit(f"unknown variant directory: {variant_root}")

    for mod in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[mod]
    _install_shims(source_root)

    eng = source_root / "vllm/parser/engine"
    _load("vllm.parser.engine.events", eng / "events.py")
    _load(
        "vllm.parser.engine.parser_engine_config",
        eng / "parser_engine_config.py",
    )
    _load("vllm.parser.engine.incremental_lexer", eng / "incremental_lexer.py")
    _load("vllm.parser.engine.token_id_scanner", eng / "token_id_scanner.py")
    _load(
        "vllm.parser.engine.streaming_parser_engine",
        variant_root / "vllm/parser/engine/streaming_parser_engine.py",
    )
    _load(
        "vllm.parser.engine.parser_engine",
        variant_root / "vllm/parser/engine/parser_engine.py",
    )
    glm47 = _load("vllm.parser.glm47_moe", source_root / "vllm/parser/glm47_moe.py")

    class Glm53Equivalent(glm47.Glm47MoeParser):
        # runtime-source/vllm/parser/glm53_moe.py:12-18
        def __init__(self, tokenizer, tools=None, **kwargs):
            kwargs = dict(kwargs)
            chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
            chat_kwargs.update(thinking=True, enable_thinking=True)
            kwargs["chat_template_kwargs"] = chat_kwargs
            super().__init__(tokenizer, tools, **kwargs)

    return Glm53Equivalent, "shim"
