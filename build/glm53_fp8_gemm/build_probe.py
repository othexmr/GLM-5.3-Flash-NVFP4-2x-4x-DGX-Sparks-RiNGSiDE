#!/usr/bin/env python3
"""Build the SM120 FP8 GEMM configs (glm53_fp8_gemm) inside the serving image and probe them (dense-fp8-kernels agent,
2026-09-24). Exploratory: it decides which configs a leaf compares; the leaf's rule is written before the leaf's data.

Build: torch.utils.cpp_extension.load (ninja, nvcc 13.0 of the image, TORCH_CUDA_ARCH_LIST=12.1a, no fast math, as
vLLM builds its SM120 scaled_mm sources), CUTLASS headers of the image (flashinfer/data/cutlass, v4.5.0 b46b16d0;
vLLM 0.29.0 pins v4.4.2, which the image does not carry: the control config tells whether the kernel differs), vLLM's
epilogue headers verbatim (vendor/). Outputs the extension, the compile log and describe() of every config.
Probe: served shapes (N x K per TP4 rank) x rows, each config x swizzle {1, 2, 4, 8} x raster {heuristic, M, N}
against vLLM's `_C.cutlass_scaled_mm` (the served kernel) on the same quantized operands: bit equality of every
output, time per call (6 rotating weight copies, a 64 MB L2 flush before each call, median of 5 x 6 calls; the
reference is timed before and after each block).
Usage: build_probe.py --src /src --out /out"""
import argparse, hashlib, json, os, statistics, sys, time, traceback
from pathlib import Path

# The image's CUTLASS copies nearest to vLLM 0.29.0's pinned v4.4.2, tried in this order (the first that builds is used
# and recorded): flashinfer's v4.5.0 (b46b16d0), vLLM's fmha_sm100 v4.3.4.
CUTLASSES = ['/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass',
             '/usr/local/lib/python3.12/dist-packages/vllm/third_party/fmha_sm100/cutlass']
SHAPES = dict(kda_in=(6416, 4096), kda_o_proj=(4096, 2048), mla_a=(2048, 4096), mla_qb=(4096, 1536),
              mla_o_proj=(4096, 4096), shared_gate_up=(1024, 4096), shared_down=(4096, 512))
ROWS = (13824, 4608, 3456, 2304, 512)
SWIZZLES = (1, 2, 4, 8)
RASTERS = (0, 1, 2)
COPIES, REPEATS = 6, 5


def build(src, out, res):
    import torch
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1a'); os.environ.setdefault('MAX_JOBS', '6')
    cfgs = sorted(p.name for p in src.glob('cfg_*.cu'))
    t0 = time.time(); attempts = []
    for i, CUTLASS in enumerate(CUTLASSES):
        bdir = out / f'build{i}'; bdir.mkdir(parents=True, exist_ok=True)
        try:
            mod = load(name='glm53_fp8_gemm', sources=[str(src / 'bind.cpp')] + [str(src / c) for c in cfgs],
                       extra_include_paths=[f'{CUTLASS}/include', f'{CUTLASS}/tools/util/include', str(src / 'vendor'), str(src)],
                       extra_cflags=['-O3', '-std=c++17'],
                       extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-DNDEBUG',
                                          '-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1', '--ptxas-options=-warn-spills',
                                          '--threads=2'],
                       extra_ldflags=['-lcuda'], build_directory=str(bdir), verbose=True)
            attempts.append(dict(cutlass=CUTLASS, ok=True)); break
        except Exception as exc:  # noqa: BLE001
            attempts.append(dict(cutlass=CUTLASS, ok=False, error=repr(exc)[-300:]))
            res['build_attempts'] = attempts
            if i == len(CUTLASSES) - 1:
                raise
    res['build_attempts'] = attempts
    so = Path(mod.__file__)
    res['build'] = dict(seconds=round(time.time() - t0, 1), so=str(so), so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
                        so_bytes=so.stat().st_size, sources={c: hashlib.sha256((src / c).read_bytes()).hexdigest()
                                                              for c in ['bind.cpp', 'glm53_fp8_gemm.cuh', *cfgs]},
                        configs=list(mod.configs()), describe=list(mod.describe()), torch=torch.__version__,
                        cutlass=CUTLASS, cutlass_version=(Path(CUTLASS) / 'include/cutlass/version.h').read_text().split(
                            'CUTLASS_MAJOR ')[1][:1] + '.' + (Path(CUTLASS) / 'include/cutlass/version.h').read_text().split(
                            'CUTLASS_MINOR ')[1][:1] + '.' + (Path(CUTLASS) / 'include/cutlass/version.h').read_text().split(
                            'CUTLASS_PATCH ')[1][:1])
    return mod


def probe(mod, out, res, save):
    import torch
    from vllm import _custom_ops as ops
    torch.manual_seed(0); dev = 'cuda'
    flush = torch.empty(16 * 1024 * 1024, device=dev, dtype=torch.float32)
    names = list(mod.configs())

    def timed(fn):
        ts = []
        for _ in range(REPEATS):
            evs = []
            for i in range(COPIES):
                flush.add_(1.0)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(i); e.record(); evs.append((s, e))
            torch.cuda.synchronize()
            ts += [s.elapsed_time(e) * 1000 for s, e in evs]
        return statistics.median(ts)

    def wq(n, k):  # the dense module's per-channel weight quantization
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        sc = (w.abs().amax(dim=1).float() / 448.0).clamp(min=1e-12)
        w8 = (w.float() / sc[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        return w8.t(), sc.reshape(n, 1).contiguous()

    res['probe'] = {}
    for fam, (n, k) in SHAPES.items():
        ws = [wq(n, k) for _ in range(COPIES)]
        for m in ROWS:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            outs = [torch.empty((m, n), device=dev, dtype=torch.bfloat16) for _ in range(COPIES)]
            ref = ops.cutlass_scaled_mm(q, ws[0][0], s, ws[0][1], torch.bfloat16)
            cell = dict(ref_us_before=timed(lambda i: ops.cutlass_scaled_mm(q, ws[i][0], s, ws[i][1], torch.bfloat16)),
                        variants={})
            for c, name in enumerate(names):
                for sw in SWIZZLES:
                    for ra in RASTERS:
                        key = f'{name}/s{sw}/r{ra}'
                        try:
                            got = torch.empty_like(ref); mod.gemm(got, q, ws[0][0], s, ws[0][1], c, sw, ra)
                            eq = bool(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
                            t = timed(lambda i: mod.gemm(outs[i], q, ws[i][0], s, ws[i][1], c, sw, ra))
                            cell['variants'][key] = dict(us=round(t, 2), bit_equal=eq)
                        except Exception as exc:  # noqa: BLE001
                            cell['variants'][key] = dict(error=repr(exc)[-300:])
            cell['ref_us_after'] = timed(lambda i: ops.cutlass_scaled_mm(q, ws[i][0], s, ws[i][1], torch.bfloat16))
            ok = {kk: v for kk, v in cell['variants'].items() if 'us' in v and v['bit_equal']}
            refm = min(cell['ref_us_before'], cell['ref_us_after'])
            if ok:
                best = min(ok, key=lambda kk: ok[kk]['us'])
                cell['best'] = dict(variant=best, us=ok[best]['us'], ratio=round(ok[best]['us'] / refm, 4))
            cell['all_bit_equal'] = all(v.get('bit_equal') for v in cell['variants'].values())
            res['probe'].setdefault(fam, {})[str(m)] = cell
            print(fam, m, 'ref', round(cell['ref_us_before'], 1), round(cell['ref_us_after'], 1), 'best', cell.get('best'),
                  'all_bit_equal', cell['all_bit_equal'], flush=True)
            save()
            del x, q, s, outs, ref
        del ws; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--src', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-probe', action='store_true'); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = dict(started=time.strftime('%F %T'), status='RUNNING')

    def save():
        tmp = a.out / 'build_probe.json.tmp'; tmp.write_text(json.dumps(res, indent=1) + '\n'); tmp.replace(a.out / 'build_probe.json')

    save()
    try:
        import torch
        torch.cuda.set_device(0)
        mod = build(a.src, a.out, res); save()
        print(json.dumps(res['build'], indent=1), flush=True)
        if not a.no_probe:
            probe(mod, a.out, res, save)
        res['status'] = 'DONE'
    except BaseException as exc:
        res.update(status='FAILED', error=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])
        raise
    finally:
        res['finished'] = time.strftime('%F %T'); save()


if __name__ == '__main__':
    sys.exit(main())
