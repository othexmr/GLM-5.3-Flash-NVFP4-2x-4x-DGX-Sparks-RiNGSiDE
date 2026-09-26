"""CPU contract tests; these do not qualify CUDA compilation or attention math."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / 'runtime-candidate'
CU = CANDIDATE / 'flashinfer/data/csrc/xqa/mha.cu'
JIT = CANDIDATE / 'flashinfer/jit/xqa.py'


class XQAPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        cls.build = Path(cls.tmp.name)
        text = CU.read_text()
        start = text.index('#if XQA_BATCH_INVARIANT_SLIDING && SLIDING_WINDOW\n// A request')
        end = text.index('void launchMHAFlashInfer(', start)
        helper = text[start:end]
        # Compile the actual candidate host policy and bounds validator. Types below
        # reproduce the retained q>1, D128 BF16 kernel's scratch layout; they do not
        # simulate its floating point attention or its CUDA synchronization.
        harness = '''#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define XQA_BATCH_INVARIANT_SLIDING 1
#define SLIDING_WINDOW 1
#define SPEC_DEC 1
constexpr bool allowMultiBlockMode = true;
constexpr struct { uint32_t x; } ctaTile{256};
constexpr uint32_t headGrpSize=4, rowsPerBlock=32, nbValidRows=32;
constexpr uint32_t gemm1WarpsPerGrp=2, nbHeadSplits=1;
struct LdGrain { char data[16]; };
struct SMemWarpRowMax { char data[128]; };
struct SharedMem { struct XSmemBuffer { static constexpr uint32_t cols=8; }; };
template <typename T, uint32_t R, uint32_t C> struct Array2D { T data[R*C]; };
template <typename T, uint32_t N> struct Vec { T data[N]; };
template <typename Offset> struct Segmenter {
  Offset offset=0;
  template <typename T> void newSeg(uint32_t count, uint32_t alignment) {
    offset = ((offset+alignment-1)/alignment)*alignment + sizeof(T)*count;
  }
  Offset getEndOffset() const { return offset; }
};
'''+helper+'''
int main(int argc, char**argv) {
 try {
  uint32_t sm=std::stoul(argv[1]), heads=std::stoul(argv[2]);
  uint32_t window=std::stoul(argv[3]), batch=std::stoul(argv[4]), q=std::stoul(argv[5]);
  if(argc>6) validateBatchInvariantSlidingWorkspace(sm,heads,window,batch,q,
          std::stoull(argv[6]),std::stoull(argv[7]));
  std::cout << batchInvariantSlidingSplitCount(sm,heads,window) << "\\n";
 } catch(std::exception const& e) { std::cerr << e.what(); return 2; }
}
'''
        cls.exe = cls.build / 'policy'
        cpp = cls.build / 'policy.cpp'
        cpp.write_text(harness)
        subprocess.run([shutil.which('clang++') or 'c++', '-std=c++17', '-O2',
                        '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_policy(self, sm=48, heads=4, window=2048, batch=1, q=5,
                   semaphore=None, scratch=None):
        args = [sm, heads, window, batch, q]
        if semaphore is not None:
            args += [semaphore, scratch]
        return subprocess.run([str(self.exe), *map(str,args)], capture_output=True, text=True)

    def test_split_count_is_stable_for_live_batch_and_graph_widths(self):
        for batch in (1,2,3,4,6,8,16,128):
            for q in (1,4,5,7,8,9,16):
                with self.subTest(batch=batch,q=q):
                    result = self.run_policy(batch=batch,q=q)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(result.stdout.strip(),'8')

    def test_retained_launch_policy_exhibits_the_regression(self):
        old = (ROOT/'original/flashinfer/data/csrc/xqa/mha.cu').read_text()
        start = old.index('void launchMHAFlashInfer(')
        launch = old[start:]
        self.assertIn('multiProcessorCount / (batchSize * nbKHeads)',launch)
        self.assertIn('divUp(maxSeqLen, ctaTile.x)',launch)
        # Exact retained formula at the live geometry, for comparison with the
        # compiled candidate: C1=12 splits, C2=6, C3=4, C4=3, C6=2.
        self.assertEqual([min(max(1,48//(b*4)),(4160+255)//256)
                          for b in (1,2,3,4,6)],[12,6,4,3,2])
        fixed = CU.read_text()[CU.read_text().index('void launchMHAFlashInfer('):]
        invariant_arm = fixed.split('#else',1)[0]
        self.assertIn('batchInvariantSlidingSplitCount(multiProcessorCount, nbKHeads, slidingWinSize)',invariant_arm)
        self.assertNotIn('batchSize * nbKHeads',invariant_arm)
        self.assertNotIn('divUp(maxSeqLen',invariant_arm)

    def test_window_and_hardware_bounds(self):
        for sm,heads,win,want in [(48,4,1,1),(48,4,256,1),(48,4,257,2),
                                 (48,4,2048,8),(48,4,4096,12),(4,8,2048,1)]:
            self.assertEqual(self.run_policy(sm,heads,win).stdout.strip(),str(want))

    def test_invalid_geometry_rejected(self):
        for kwargs in ({'heads':0},{'window':0},{'batch':0},{'q':0}):
            self.assertNotEqual(self.run_policy(**kwargs,semaphore=8388608,scratch=125829120).returncode,0)

    def test_exact_workspace_bound_and_undersized_rejection(self):
        # q<=8: one token block per KV head. Each split stores 2*128 bytes
        # row stats plus 32*128*2 bytes BF16 partial output.
        n=6*4*8
        required=n*(2*128+32*128*2)
        self.assertEqual(required,1622016)
        result=self.run_policy(batch=6,q=8,semaphore=6*4*4,scratch=required)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.run_policy(batch=6,q=8,semaphore=6*4*4,scratch=required-1).returncode,2)
        self.assertEqual(self.run_policy(batch=6,q=8,semaphore=6*4*4-1,scratch=required).returncode,2)
        self.assertEqual(self.run_policy(batch=6,q=9,semaphore=2*6*4*4,scratch=2*required).returncode,0)
        self.assertEqual(self.run_policy(batch=6,q=9,semaphore=2*6*4*4,scratch=required).returncode,2)

    def test_workspace_index_overflow_rejected(self):
        result=self.run_policy(batch=4294967295,q=8,semaphore=2**40,scratch=2**40)
        self.assertEqual(result.returncode,2)
        self.assertIn('overflow',result.stderr)

    def test_guard_runs_before_gpu_launch(self):
        wrapper=(CANDIDATE/'flashinfer/data/csrc/xqa/xqa_wrapper.cu').read_text()
        self.assertLess(wrapper.index('validateBatchInvariantSlidingWorkspace('),wrapper.index('  launchMHAFlashInfer('))


class JITIdentityTests(unittest.TestCase):
    def spec(self, arch, window, q=5, ragged=False, original=False):
        file=(ROOT/'original/flashinfer/jit/xqa.py')if original else JIT
        tree=ast.parse(file.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gen_xqa_module')
        module=ast.Module(body=[function],type_ignores=[])
        torch=types.SimpleNamespace(float16='f16',bfloat16='bf16',float8_e4m3fn='f8',int8='i8',uint8='u8',dtype=str)
        class Context:
            TARGET_CUDA_ARCHS=[(arch,1 if arch==12 else 0)]
            def get_nvcc_flags_list(self,**kwargs):return ['-arch=sm_121'if arch==12 else'-arch=sm_90']
        capture=lambda name,sources,**kwargs:dict(name=name,sources=sources,**kwargs)
        env={'torch':torch,'JitSpec':dict,'CompilationContext':Context,
             '_has_sm90_target':lambda:arch==9,'swap_ab_eligible':lambda q,h:q*h<=32,
             'ragged_q_changes_build':lambda q,h:arch==9 and q*h<=32,
             'jit_env':types.SimpleNamespace(FLASHINFER_CSRC_DIR=Path('/csrc')),
             'gen_jit_spec':capture,'filename_safe_dtype_map':{'bf16':'bf16','f16':'f16','f8':'f8'},
             'xqa_nvcc_flags':['-DNDEBUG=1']}
        exec(compile(module,str(file),'exec'),env)
        return env['gen_xqa_module']('bf16','bf16',64,128,4,window,'bf16',q,ragged)

    def test_sm12_sliding_cannot_reuse_old_jit_identity(self):
        for q in (1,5,8):
            for ragged in (False,True)if q>1 else(False,):
                fixed=self.spec(12,True,q,ragged)
                old=self.spec(12,True,q,ragged,original=True)
                self.assertEqual(fixed['name'],old['name']+'_batch_invariant_sliding_v1')
                self.assertIn('-DXQA_BATCH_INVARIANT_SLIDING=1',fixed['extra_cuda_cflags'])

    def test_full_attention_and_non_sm12_unchanged(self):
        for arch,win in ((12,False),(9,False),(9,True),(10,True)):
            self.assertEqual(self.spec(arch,win),self.spec(arch,win,original=True))


class InstallTests(unittest.TestCase):
    def test_manifest_matches_all_sources(self):
        for row in json.loads((ROOT/'source-manifest.json').read_text()):
            for directory,key in [('original','before'),('runtime-candidate','after')]:
                self.assertEqual(hashlib.sha256((ROOT/directory/row['path']).read_bytes()).hexdigest(),row[key])

    def test_install_idempotence_and_complete_prevalidation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            site=Path(tmp)/'site'
            shutil.copytree(ROOT/'original',site)
            script=[sys.executable,str(ROOT/'install.py'),'--site',str(site)]
            subprocess.run(script,check=True,capture_output=True)
            subprocess.run(script,check=True,capture_output=True)
            rows=json.loads((ROOT/'source-manifest.json').read_text())
            for row in rows:
                self.assertEqual(hashlib.sha256((site/row['path']).read_bytes()).hexdigest(),row['after'])
            shutil.rmtree(site);shutil.copytree(ROOT/'original',site)
            drift=site/rows[-1]['path'];drift.write_text(drift.read_text()+'\n# drift\n')
            self.assertNotEqual(subprocess.run(script,capture_output=True).returncode,0)
            self.assertEqual(hashlib.sha256((site/rows[0]['path']).read_bytes()).hexdigest(),rows[0]['before'])


if __name__=='__main__':
    unittest.main(verbosity=2)
