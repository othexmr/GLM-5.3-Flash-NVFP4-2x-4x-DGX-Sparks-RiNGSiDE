"""UTF-8 continuation guard for the V2 sampler (topp-fix, 2026-09-25; GLM53_UTF8_GUARD=1, default off), with optional
folding (GLM53_UTF8_GUARD_FOLD=1).

Byte-level BPE can end a token inside a UTF-8 character (a lead byte, or a lead byte and some continuation bytes).
The next sampled token must then continue that character. When the model instead samples a token that starts a new
character (an observed "one-character glitch"), the detokenizer emits U+FFFD.

What the guard does:
- Scope: every logits row of a sampled request (temperature > 0) whose sequence ends inside a character. Each token
  that is not a valid continuation under a strict UTF-8 decoder gets -inf.
- Placement: Sampler.apply_sampling_params, after the thinking-budget forcing and temperature, before min_p,
  top-k/top-p and sampling. Masking commutes with temperature; folding has to see the tempered logits.
- Speculative verification: the same function serves the target rows. There a row's state includes the drafted tokens
  before it, so a drafted invalid or folded token has target probability 0, and recovered and bonus tokens come from
  the transformed distribution. Token ids are never substituted after acceptance.
- Other rows: not touched (bit-identical). GLM53_UTF8_GUARD=2 also masks, at a boundary, the tokens that are invalid
  there (the 256 byte-invalid tokens of this vocabulary); that mode is not bit-identical.

Folding (review 4, relayed 2026-09-25):
- The "mixed" tokens are 44421 = A4 ED, 56776 = A0 ED, 71940 = 92 E1 9E, 106890 = AB E7 97 and 129292 = B1 DB. Each
  completes a pending character with its first byte and opens another one.
- At the state where that is legal (one byte missing), the mixed token's probability moves to its completing prefix
  (97, 254, 240, 104, 109): L[prefix] = logaddexp(L[prefix], L[mixed]), then L[mixed] = -inf.
- The completed character is kept, and the next step chooses the next character fresh. The mask alone cannot undo an
  already-chosen ED: after it, only ED-range characters can follow.
- The pairs are derived from the table at start (fold_pairs) and logged.

States (the pending prefix decides the range of the next byte, strict UTF-8, RFC 3629):
- 0 clean; 1 one byte missing; 2 two missing (80-BF); 3 two missing after E0 (A0-BF); 4 two missing after ED
  (80-9F, no surrogates); 5 three missing (80-BF); 6 three missing after F0 (90-BF); 7 three missing after F4 (80-8F).
- Later continuation bytes are 80-BF.
- A token is valid at a pending state when its bytes, fed to the strict decoder from that state, raise no error. The
  token may finish the character and continue with valid text, or end inside a (new or the same) character.

Termination:
- Special tokens (added tokens marked special) have no bytes, and neither do padding ids beyond the tokenizer.
  Non-special added tokens (<think>, </think>, <tool_call>, ...) are their ASCII text. None of these continues a
  character, so all are masked at a pending state: a stop token or an end-of-thinking token waits until the
  character completes.
- With folding on, that is at most three more tokens: no token can then complete a character and open another, so
  every step lowers the missing count. Without folding, a mixed token can open a new character, and the pending run can
  go on (each character still completes within three bytes).
- The thinking-budget forcing is a boost that keeps the other logits. The guard runs after it, so the forced marker
  waits too.
- max_tokens, a stop string or a client abort can still end a response inside a character. That needs a look-ahead
  the sampler does not have.

Other behaviour:
- Greedy requests (temperature 0) are not guarded by default. Their batches can skip logits processing entirely: the
  greedy argmax verification (GLM53_GREEDY_ARGMAX_VERIFY) needs that, and RigMark runs greedy. Guarding greedy rows only
  in mixed batches would make greedy output depend on batch composition. GLM53_UTF8_GUARD_GREEDY=1 guards them too and
  marks greedy requests as needing processing, so every batch then takes the guarded path.
- Mode 2 at a boundary writes -inf to the fixed list of byte-invalid tokens (256 for this vocabulary) with one scatter.
  A request whose logits have a pending row is handled as in mode 1.
- Empty support: a row whose valid tokens all have -inf logits (another constraint, e.g. a grammar) is left as the
  other constraints made it. The guard never unmasks a token and never empties a row. The row is reported (state -s,
  counted on the device, and logged as a WARNING every 256 calls while the count grows:
  detected, not silent.
- The guard prevents malformed bytes; it cannot make the model choose the character it meant.

The table:
- Built once per process from the served tokenizer (GLM53_UTF8_GUARD_TOKENIZER, default
  /models/target/tokenizer.json).
- Per token: its bytes (byte-level BPE alphabet), the validity bit per state, and the next state from each of the
  8 states.
- The state after the committed sequence is the strict decoder's state after its last 4 tokens: a pending character
  spans at most 4 bytes, and each token has at least one byte.
- Invalid bytes resynchronise the way Python's decoder does."""
import hashlib
import json
import os

# 1: pending rows only (rows at a character boundary are untouched); 2: also mask, at a boundary, the tokens that
# are invalid there (a stray continuation byte, C0/C1/F5-FF, a broken sequence inside the token).
MODE = {"1": 1, "2": 2}.get(os.environ.get("GLM53_UTF8_GUARD", "").strip(), 0)
ENABLED = MODE > 0
# GLM53_UTF8_GUARD_FOLD=1 (with the guard): at a pending row where a "mixed" token legally completes the character and
# opens another one, its probability moves to its completing prefix token and it is banned (see fold_pairs).
FOLD = os.environ.get("GLM53_UTF8_GUARD_FOLD", "").strip() == "1"
# GLM53_UTF8_GUARD_GREEDY=1: guard greedy (temperature 0) requests too; they then need logits processing, which takes
# them off the greedy argmax verification path (GLM53_GREEDY_ARGMAX_VERIFY), so it is off by default.
GREEDY = os.environ.get("GLM53_UTF8_GUARD_GREEDY", "").strip() == "1"
TOKENIZER = os.environ.get("GLM53_UTF8_GUARD_TOKENIZER", "").strip() or "/models/target/tokenizer.json"
N_STATES = 8
_RANGES = {1: (0x80, 0xBF), 2: (0x80, 0xBF), 3: (0xA0, 0xBF), 4: (0x80, 0x9F), 5: (0x80, 0xBF), 6: (0x90, 0xBF),
           7: (0x80, 0x8F)}
_MISSING = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}


def _lead_state(b):
    """State after a byte fed at state 0; None when the byte cannot start a character."""
    if b < 0x80:
        return 0
    if 0xC2 <= b <= 0xDF:
        return 1
    if b == 0xE0:
        return 3
    if b == 0xED:
        return 4
    if 0xE1 <= b <= 0xEF:
        return 2
    if b == 0xF0:
        return 6
    if 0xF1 <= b <= 0xF3:
        return 5
    if b == 0xF4:
        return 7
    return None


def strict_step(state, b):
    """One byte through the strict decoder: the new state, or None on an error."""
    if state == 0:
        return _lead_state(b)
    lo, hi = _RANGES[state]
    if not (lo <= b <= hi):
        return None
    k = _MISSING[state] - 1
    return {0: 0, 1: 1, 2: 2}[k]


def feed(state, b):
    """One byte with resynchronisation (Python's decoder: the broken sequence becomes U+FFFD, the byte is re-read)."""
    r = strict_step(state, b)
    if r is not None:
        return r
    if state != 0:
        r = strict_step(0, b)
        return 0 if r is None else r
    return 0


def valid_from(state, data):
    """Whether the bytes continue a pending state without a decoder error (the end may be inside a character)."""
    for b in data:
        state = strict_step(state, b)
        if state is None:
            return False
    return True


def next_state(state, data):
    for b in data:
        state = feed(state, b)
    return state


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_bytes(tokenizer_path, vocab_size):
    """Per id: the token's bytes, or None when it carries no text bytes of its own (special, padding)."""
    tok = json.loads(open(tokenizer_path, encoding="utf-8").read())
    dec = {v: k for k, v in _bytes_to_unicode().items()}
    out = [None] * vocab_size
    for s, i in tok["model"]["vocab"].items():
        if i < vocab_size:
            try:
                out[i] = bytes(dec[c] for c in s)
            except KeyError:
                out[i] = None
    for a in tok.get("added_tokens", []):
        i = a["id"]
        if i < vocab_size:
            out[i] = None if a.get("special") else a["content"].encode("utf-8")
    return out


def build_table(tokenizer_path, vocab_size):
    """(valid_bits [V] uint8: bit s set when valid at state s (bit 0: from a boundary; tokens without bytes are
    valid there and invalid at every pending state); next [8, V] int8; info dict)."""
    import numpy as np

    tb = token_bytes(tokenizer_path, vocab_size)
    valid = np.zeros(vocab_size, np.uint8)
    nxt = np.zeros((N_STATES, vocab_size), np.int8)
    for i, data in enumerate(tb):
        if data is None:
            valid[i] = 1   # no bytes: fine at a boundary, invalid at every pending state; resets to 0
            continue
        bits = 0
        for s in range(0, N_STATES):
            if valid_from(s, data):
                bits |= 1 << s
        valid[i] = bits
        for s in range(N_STATES):
            nxt[s, i] = next_state(s, data)
    h = hashlib.sha256(valid.tobytes() + nxt.tobytes()).hexdigest()
    info = dict(tokenizer=tokenizer_path, tokenizer_sha256=hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                vocab_size=vocab_size, table_sha256=h, tokens_without_bytes=sum(1 for d in tb if d is None),
                valid_counts={s: int(((valid >> s) & 1).sum()) for s in range(0, N_STATES)})
    return valid, nxt, info


def fold_pairs(tokenizer_path, vocab_size, valid, nxt):
    """(mixed, prefix, state) for every token that is valid at a pending state s, completes the character with its
    first bytes and opens another character (its next state is pending again), and whose completing bytes are a token
    of their own (the prefix). For this vocabulary: 44421->97 (A4), 56776->254 (A0), 71940->240 (92), 106890->104
    (AB), 129292->109 (B1), all at state 1 (review 4, relayed 2026-09-25)."""
    tb = token_bytes(tokenizer_path, vocab_size)
    byid = {}
    for i, d in enumerate(tb):
        if d is not None and d not in byid:
            byid[d] = i
    out = []
    for i, d in enumerate(tb):
        if not d:
            continue
        k = next((j for j, b in enumerate(d) if not 0x80 <= b <= 0xBF), len(d))
        if 0 < k < len(d):
            for s in range(1, N_STATES):
                if (int(valid[i]) >> s) & 1 and _MISSING[s] == k and int(nxt[s, i]) != 0 and d[:k] in byid:
                    out.append((i, byid[d[:k]], s))
    return out


_GUARD = None


def get_guard(vocab_size, device):
    global _GUARD
    if _GUARD is None:
        _GUARD = Utf8Guard(vocab_size, device)
    return _GUARD


try:  # the kernel needs the serving image (triton); the table and the CPU tests do not
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _utf8_state_kernel(
        LOGITS,
        LOGITS_STRIDE,
        EXPANDED_IDX_MAPPING,
        ALL_TOKEN_IDS,
        ALL_TOKEN_IDS_STRIDE,
        TOTAL_LEN,
        INPUT_IDS,
        EXPANDED_LOCAL_POS,
        TEMPERATURE,
        NEXT,
        VALID_IDS,
        VALID_N,
        STATES_OUT,
        MASK_CODE,
        FOLD_MIXED,
        FOLD_PREFIX,
        FOLD_STATE,
        CONFLICTS,
        VOCAB_SIZE: tl.constexpr,
        MAX_DRAFTS: tl.constexpr,
        MODE: tl.constexpr,
        N_FOLD: tl.constexpr,
        GREEDY: tl.constexpr,
        MAXV: tl.constexpr,
    ):
        """Per row: the state (last 4 committed tokens + the drafts before the row), the support check over the state's
        valid ids (a gather of at most MAXV logits), and folding. Writes the row's audit state and its mask code for
        _utf8_mask_kernel (0 nothing, 1..7 the pending state, 8 a boundary row in mode 2)."""
        row = tl.program_id(0).to(tl.int64)
        req = tl.load(EXPANDED_IDX_MAPPING + row).to(tl.int64)
        j = tl.load(EXPANDED_LOCAL_POS + row).to(tl.int64)
        total = tl.load(TOTAL_LEN + req).to(tl.int64)
        temp = tl.load(TEMPERATURE + req).to(tl.float32)
        s = tl.zeros((), dtype=tl.int32)
        for i in range(4):
            pos = total - 4 + i
            if pos >= 0:
                tok = tl.load(ALL_TOKEN_IDS + req * ALL_TOKEN_IDS_STRIDE + pos).to(tl.int64)
                ok = (tok >= 0) & (tok < VOCAB_SIZE)
                s = tl.where(ok, tl.load(NEXT + s.to(tl.int64) * VOCAB_SIZE + tl.where(ok, tok, 0)).to(tl.int32), 0)
        first = row - j
        for i in range(1, MAX_DRAFTS + 1):
            if i <= j:
                tok = tl.load(INPUT_IDS + first + i).to(tl.int64)
                ok = (tok >= 0) & (tok < VOCAB_SIZE)
                s = tl.where(ok, tl.load(NEXT + s.to(tl.int64) * VOCAB_SIZE + tl.where(ok, tok, 0)).to(tl.int32), 0)
        if GREEDY:
            out = s
            guarded = temp == temp
        else:
            out = tl.where(temp == 0.0, 0, s)
            guarded = temp != 0.0
        code = tl.zeros((), dtype=tl.int32)
        if MODE == 2:
            code = tl.where(guarded & (out == 0), 8, 0)
        folded = tl.zeros((), dtype=tl.int32)
        if out != 0:
            # support: some valid continuation must have a finite logit (read before folding; folding keeps it true)
            nv = tl.load(VALID_N + out)
            o = tl.arange(0, MAXV)
            vid = tl.load(VALID_IDS + out * MAXV + o, mask=o < nv, other=0).to(tl.int64)
            xv = tl.load(LOGITS + row * LOGITS_STRIDE + vid, mask=o < nv, other=float("-inf"))
            if tl.max(xv) > float("-inf"):
                code = out
                # folding: a mixed token's probability moves to its completing prefix, the mixed token is banned
                for f in range(N_FOLD):
                    if tl.load(FOLD_STATE + f) == out:
                        mt = tl.load(FOLD_MIXED + f).to(tl.int64)
                        pt = tl.load(FOLD_PREFIX + f).to(tl.int64)
                        lm = tl.load(LOGITS + row * LOGITS_STRIDE + mt)
                        lp = tl.load(LOGITS + row * LOGITS_STRIDE + pt)
                        if lm > float("-inf"):
                            mx = tl.maximum(lm, lp)
                            tl.store(LOGITS + row * LOGITS_STRIDE + pt, mx + tl.log(tl.exp(lm - mx) + tl.exp(lp - mx)))
                            tl.store(LOGITS + row * LOGITS_STRIDE + mt, float("-inf"))
                            folded = tl.full((), 16, tl.int32)
            else:
                # every valid continuation is -inf (another constraint): left unmasked, reported and counted
                out = -out
                tl.atomic_add(CONFLICTS, 1)
        tl.store(STATES_OUT + row, tl.where(out > 0, out | folded, out))
        tl.store(MASK_CODE + row, code)

    @triton.jit
    def _utf8_mask_kernel(
        LOGITS,
        LOGITS_STRIDE,
        MASK_CODE,
        VALID,
        BINV,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        N_BINV: tl.constexpr,
        BINV_BLOCK: tl.constexpr,
    ):
        """Grid (rows, vocab blocks): a pending row's invalid tokens in this block become -inf; a mode-2 boundary row
        gets -inf on the byte-invalid tokens (block 0, one scatter). Valid tokens are never written."""
        row = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1)
        code = tl.load(MASK_CODE + row)
        if (code > 0) & (code < 8):
            offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            msk = offs < VOCAB_SIZE
            bits = tl.load(VALID + offs, mask=msk, other=255).to(tl.int32)
            tl.store(LOGITS + row * LOGITS_STRIDE + offs, float("-inf"), mask=msk & (((bits >> code) & 1) == 0))
        if (code == 8) & (blk == 0):
            ob = tl.arange(0, BINV_BLOCK)
            bid = tl.load(BINV + ob, mask=ob < N_BINV, other=0).to(tl.int64)
            tl.store(LOGITS + row * LOGITS_STRIDE + bid, float("-inf"), mask=ob < N_BINV)
except Exception:  # pragma: no cover - CPU-only environments
    _utf8_state_kernel = _utf8_mask_kernel = None


class Utf8Guard:
    """The table on the device and the per-step launch; last_states holds the rows' states of the last call."""

    def __init__(self, vocab_size, device):
        import torch

        valid, nxt, info = build_table(TOKENIZER, vocab_size)
        self.vocab_size = vocab_size
        self.valid = torch.from_numpy(valid).to(device)
        self.next = torch.from_numpy(nxt).to(device)
        pairs = fold_pairs(TOKENIZER, vocab_size, valid, nxt) if FOLD else []
        info = dict(info, fold=FOLD, fold_pairs=[list(x) for x in pairs])
        self.n_fold = max(1, len(pairs))
        z = pairs or [(0, 0, -1)]
        self.fold_mixed = torch.tensor([x[0] for x in z], dtype=torch.int32, device=device)
        self.fold_prefix = torch.tensor([x[1] for x in z], dtype=torch.int32, device=device)
        self.fold_state = torch.tensor([x[2] for x in z], dtype=torch.int32, device=device)
        self.conflicts = torch.zeros(1, dtype=torch.int32, device=device)
        self.maxv = 256
        ids = [[i for i in range(vocab_size) if (int(valid[i]) >> st) & 1] if st else [] for st in range(N_STATES)]
        assert all(len(x) <= self.maxv for x in ids), [len(x) for x in ids]
        self.valid_ids = torch.tensor([x + [0] * (self.maxv - len(x)) for x in ids], dtype=torch.int32, device=device)
        self.valid_n = torch.tensor([len(x) for x in ids], dtype=torch.int32, device=device)
        binv = [i for i in range(vocab_size) if not (int(valid[i]) & 1)]
        self.n_binv = max(1, len(binv))
        self.binv = torch.tensor(binv or [0], dtype=torch.int32, device=device)
        self.mode = MODE
        self.greedy = GREEDY
        info = dict(info, greedy=GREEDY, boundary_invalid=len(binv))
        self.calls = 0
        self.reported = 0
        self.info = info
        self.last_states = None
        print("GLM53_UTF8_GUARD engaged mode=%d " % MODE + json.dumps(info, sort_keys=True), flush=True)

    def apply(self, logits, expanded_idx_mapping, all_token_ids, total_len, input_ids, expanded_local_pos,
              temperature, max_drafts=16):
        import torch

        n = logits.shape[0]
        states = torch.empty(n, dtype=torch.int32, device=logits.device)
        code = torch.empty(n, dtype=torch.int32, device=logits.device)
        if n:
            _utf8_state_kernel[(n,)](
                logits, logits.stride(0), expanded_idx_mapping, all_token_ids, all_token_ids.stride(0), total_len,
                input_ids, expanded_local_pos, temperature, self.next, self.valid_ids, self.valid_n, states, code,
                self.fold_mixed, self.fold_prefix, self.fold_state, self.conflicts,
                VOCAB_SIZE=self.vocab_size, MAX_DRAFTS=max_drafts, MODE=self.mode, N_FOLD=self.n_fold,
                GREEDY=int(self.greedy), MAXV=self.maxv, num_warps=1)
            _utf8_mask_kernel[(n, triton.cdiv(self.vocab_size, 8192))](
                logits, logits.stride(0), code, self.valid, self.binv,
                VOCAB_SIZE=self.vocab_size, BLOCK_SIZE=8192, N_BINV=self.n_binv,
                BINV_BLOCK=triton.next_power_of_2(self.n_binv), num_warps=8)
        self.last_states = states
        self.calls += 1
        if self.calls % 256 == 0:   # one sync per 256 steps: an empty-support conflict is logged, never silent
            c = int(self.conflicts.item())
            if c > self.reported:
                print("GLM53_UTF8_GUARD WARNING empty_support rows=%d (left unmasked: another constraint removed every "
                      "valid continuation)" % c, flush=True)
                self.reported = c
        return states
