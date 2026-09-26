from pathlib import Path
import difflib, hashlib, json
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / 'nvfp4-workflows/runtime-source'
# Resolve from work/, independent of caller cwd.
BASE = ROOT.parents[1] / 'nvfp4-workflows/runtime-source'
FILES = ['flashinfer/data/csrc/xqa/mha.cu', 'flashinfer/data/csrc/xqa/mha.h', 'flashinfer/data/csrc/xqa/xqa_wrapper.cu', 'flashinfer/jit/xqa.py']
source = {p: (BASE / p).read_text() for p in FILES}
def replace(path, old, new):
    if source[path].count(old) != 1:
        raise RuntimeError(f'anchor count changed: {path}: {old[:80]}')
    source[path] = source[path].replace(old, new)
cu = FILES[0]
helper = '''#if XQA_BATCH_INVARIANT_SLIDING && SLIDING_WINDOW
// A request must keep the same partial-softmax partition when other requests
// join the batch or graph padding / page-table capacity changes. Retain bounded
// split-K parallelism, sized by the configured window and hardware only.
static uint32_t batchInvariantSlidingSplitCount(uint32_t multiProcessorCount,
                                                uint32_t nbKHeads,
                                                uint32_t slidingWinSize) {
  if (nbKHeads == 0 || slidingWinSize == 0) {
    throw std::runtime_error("batch-invariant XQA requires KV heads and a positive window");
  }
  if (!allowMultiBlockMode) {
    return 1;
  }
  uint32_t const windowTiles = slidingWinSize / ctaTile.x +
                               uint32_t(slidingWinSize % ctaTile.x != 0);
  return std::min<uint32_t>(std::max<uint32_t>(1U, multiProcessorCount / nbKHeads),
                            windowTiles);
}

// Fixed per-request splitting grows scratch with the batch, unlike the old
// occupancy heuristic. Check the exact kernel layout before launching; never
// shrink the split count to fit scratch because that would change the math.
void validateBatchInvariantSlidingWorkspace(uint32_t multiProcessorCount, uint32_t nbKHeads,
                                            uint32_t slidingWinSize, uint32_t batchSize,
                                            uint32_t qSeqLen, uint64_t semaphoreBytes,
                                            uint64_t scratchBytes) {
  uint32_t const splits =
      batchInvariantSlidingSplitCount(multiProcessorCount, nbKHeads, slidingWinSize);
  if (batchSize == 0 || qSeqLen == 0) {
    throw std::runtime_error("batch-invariant XQA requires a nonempty batch and query width");
  }
  if (splits == 1) {
    return;
  }
  uint64_t nbSeq = uint64_t(batchSize) * nbKHeads;
#if SPEC_DEC
  uint64_t const headTokens = uint64_t(qSeqLen) * headGrpSize;
  uint64_t const tokenBlocks = headTokens / rowsPerBlock + (headTokens % rowsPerBlock != 0);
  if (tokenBlocks > 0xffffffffULL / nbSeq) {
    throw std::runtime_error("batch-invariant XQA workspace index overflow");
  }
  nbSeq *= tokenBlocks;
#endif
  if (nbSeq > 0xffffffffULL / splits) {
    throw std::runtime_error("batch-invariant XQA workspace index overflow");
  }
  uint32_t const nbSubSeq = uint32_t(nbSeq * splits);
  using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
  using PartialTile = Vec<ScratchBuf, gemm1WarpsPerGrp * nbHeadSplits>;
  Segmenter<uint64_t> layout;
  layout.newSeg<SMemWarpRowMax>(nbSubSeq, sizeof(SMemWarpRowMax));
  layout.newSeg<SMemWarpRowMax>(nbSubSeq, sizeof(SMemWarpRowMax));
  layout.newSeg<PartialTile>(nbSubSeq, sizeof(PartialTile));
  uint64_t const requiredScratch = layout.getEndOffset();
  if (requiredScratch > 0xffffffffULL || scratchBytes < requiredScratch ||
      semaphoreBytes < nbSeq * sizeof(uint32_t)) {
    throw std::runtime_error("batch-invariant XQA workspace is too small for fixed sliding splits");
  }
}
#endif

'''
replace(cu, '#include "cuda_hint.cuh"', '#include <stdexcept>\n\n#include "cuda_hint.cuh"')
replace(cu, 'void launchMHAFlashInfer(uint32_t multiProcessorCount,', helper + 'void launchMHAFlashInfer(uint32_t multiProcessorCount,')
replace(cu, '''  uint32_t const nbSubSeqPerSeq = [&]() -> uint32_t {
    if (!allowMultiBlockMode) {
      return 1;
    }
    return std::min<uint32_t>(std::max<uint32_t>(1U, multiProcessorCount / (batchSize * nbKHeads)),
                              divUp(maxSeqLen, ctaTile.x));
  }();''', '''  uint32_t const nbSubSeqPerSeq = [&]() -> uint32_t {
#if XQA_BATCH_INVARIANT_SLIDING && SLIDING_WINDOW
    return batchInvariantSlidingSplitCount(multiProcessorCount, nbKHeads, slidingWinSize);
#else
    if (!allowMultiBlockMode) {
      return 1;
    }
    return std::min<uint32_t>(std::max<uint32_t>(1U, multiProcessorCount / (batchSize * nbKHeads)),
                              divUp(maxSeqLen, ctaTile.x));
#endif
  }();''')
replace(FILES[1], 'void launchMHA(\n', '''#if XQA_BATCH_INVARIANT_SLIDING && SLIDING_WINDOW
void validateBatchInvariantSlidingWorkspace(uint32_t multiProcessorCount, uint32_t nbKHeads,
                                            uint32_t slidingWinSize, uint32_t batchSize,
                                            uint32_t qSeqLen, uint64_t semaphoreBytes,
                                            uint64_t scratchBytes);
#endif

void launchMHA(
''')
replace(FILES[2], '  launchMHAFlashInfer(multiProcessorCount, nbKHeads, slidingWinSize, qScale, qScalePtr,', '''#if XQA_BATCH_INVARIANT_SLIDING && SLIDING_WINDOW
  validateBatchInvariantSlidingWorkspace(
      multiProcessorCount, nbKHeads, slidingWinSize, batchSize, qSeqLen,
      uint64_t(semaphores.numel()) * get_element_size(semaphores),
      uint64_t(scratch.numel()) * get_element_size(scratch));
#endif

  launchMHAFlashInfer(multiProcessorCount, nbKHeads, slidingWinSize, qScale, qScalePtr,''')
replace(FILES[3], '''    # Suffix the URI only when ragged Q actually changes the compile flags
''', '''    # SM12x DFlash sliding attention uses a stable per-request split count.
    # Version the JIT URI as well as the flags: existing prebuilt/cached modules
    # must never satisfy this new math policy under their old identity.
    batch_invariant_sliding = use_sliding_window and any(
        major == 12 for major, _ in compilation_context.TARGET_CUDA_ARCHS
    )
    sliding_suffix = "_batch_invariant_sliding_v1" if batch_invariant_sliding else ""
    flag_batch_invariant_sliding = (
        ["-DXQA_BATCH_INVARIANT_SLIDING=1"] if batch_invariant_sliding else []
    )

    # Suffix the URI only when ragged Q actually changes the compile flags
''')
replace(FILES[3], '{ragged_suffix}",\n        sources,', '{ragged_suffix}{sliding_suffix}",\n        sources,')
replace(FILES[3], '        + flag_sm90_mha,', '        + flag_sm90_mha\n        + flag_batch_invariant_sliding,')
manifest = []
diffs = []
for p, fixed in source.items():
    before = (BASE / p).read_bytes()
    for d, content in [('original', before), ('runtime-candidate', fixed.encode())]:
        dest = ROOT / d / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    manifest.append({'path': p, 'before': hashlib.sha256(before).hexdigest(), 'after': hashlib.sha256(fixed.encode()).hexdigest()})
    diffs.extend(difflib.unified_diff(before.decode().splitlines(True), fixed.splitlines(True), fromfile='a/'+p, tofile='b/'+p))
(ROOT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(ROOT / 'drafter-batch-fix.patch').write_text(''.join(diffs))
print(json.dumps(manifest, indent=2))
