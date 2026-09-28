# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of generated switched source and the opt-in launcher.

The native tests compile the actual host-side selector/mapping functions with a
small verbs-table fixture. They do not load CUDA, libibverbs, or open a device.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load('switched_prepare_test', 'launch/options/switched-roce/prepare.py')
render = load('switched_render_test', 'launch/render.py')
compose = load('switched_compose_test', 'launch/compose.py')


def protocol(source, name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def c_function(source, name):
    start = source.index('static int ' + name + '(')
    begin = source.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


def fixture(topo='tp4'):
    root = ROOT / 'launch/profiles' / topo
    return json.loads((root / 'profile.json').read_text()), json.loads((root / 'canonical-launch.json').read_text())


class SwitchedOption(unittest.TestCase):
    def test_pins_reproducibility_and_unchanged_default_mapping(self):
        derived = prepare.derive()
        for rel, value in prepare.changes().items():
            self.assertEqual(prepare.sha(derived[rel]), value['sha256'])
        with patch.dict(os.environ, {'GLM53_RDMA_SWITCHED': '0'}):
            for package in ('glm53_lean_allreduce', 'glm53_prefill_rdma'):
                rel = package + '/protocol.py'
                old = protocol((ROOT / 'src/tp4' / rel).read_bytes(), package + '_old')
                new = protocol(derived[rel], package + '_new')
                for rank in range(4):
                    if 'lean' in package:
                        for phase in (1, 2):
                            self.assertEqual(new.pair_map(rank, phase), old.pair_map(rank, phase))
                    else:
                        for peer in ((rank + 1) % 4, (rank - 1) % 4):
                            for lane in (0, 1):
                                self.assertEqual(new.direct_path(rank, peer, lane), old.direct_path(rank, peer, lane))

    def test_switched_peer_lane_mapping_and_bad_edges(self):
        files = prepare.derive()
        with patch.dict(os.environ, {'GLM53_RDMA_SWITCHED': '1'}):
            lean = protocol(files['glm53_lean_allreduce/protocol.py'], 'lean_switched')
            prefill = protocol(files['glm53_prefill_rdma/protocol.py'], 'prefill_switched')
            for rank in range(4):
                for phase in (1, 2):
                    self.assertEqual(sorted(lean.pair_map(rank, phase)), [(), (0, 1)])
                for peer in ((rank + 1) % 4, (rank - 1) % 4):
                    self.assertEqual([prefill.direct_path(rank, peer, x) for x in (0, 1)], [1, 3])
            for args in [(0, 2, 0), (0, 0, 0), (-1, 0, 0), (4, 1, 0), (0, 1, 2)]:
                with self.assertRaises(ValueError):
                    prefill.direct_path(*args)

    def test_default_render_is_identical_and_flag_is_explicit(self):
        for topo in ('tp4', 'tp2'):
            prof, canonical = fixture(topo)
            for rank, argv in canonical['argv'].items():
                self.assertEqual(render.render(prof, canonical['site'], int(rank)), argv)
                self.assertEqual(render.render(prof, dict(canonical['site'], GLM53_RDMA_SWITCHED='0'), int(rank)), argv)
        prof, canonical = fixture()
        for invalid in ['yes', '-1']:
            with self.assertRaises(ValueError):
                render.render(prof, dict(canonical['site'], GLM53_RDMA_SWITCHED=invalid), 0)
        with self.assertRaises(ValueError):
            render.render(prof, dict(canonical['site'], GLM53_RDMA_SWITCHED='1'), 0)
        prof2, canonical2 = fixture('tp2')
        with self.assertRaises(ValueError):
            render.render(prof2, dict(canonical2['site'], GLM53_RDMA_SWITCHED='1',
                                    SWITCHLESS_SWITCHED_ROCE_DIR='/candidate'), 0)

    def test_option_changes_only_seven_mounts_and_three_environment_entries(self):
        prof, canonical = fixture()
        site = dict(canonical['site'], GLM53_RDMA_SWITCHED='1', SWITCHLESS_SWITCHED_ROCE_DIR='/candidate')
        for rank, original in canonical['argv'].items():
            argv = render.render(prof, site, int(rank))
            stripped, i, mounts = [], 0, []
            while i < len(argv):
                if argv[i] == '--mount' and 'src=/candidate/' in argv[i + 1]:
                    mounts.append(argv[i + 1]); i += 2; continue
                if argv[i:i + 2] == ['-e', 'GLM53_RDMA_SWITCHED=1']:
                    i += 2; continue
                stripped.append(argv[i]); i += 1
            expected, i = [], 0
            while i < len(original):
                if original[i] == '-e' and original[i + 1].split('=', 1)[0] in ('NCCL_SWITCHLESS_RING_ONLY', 'NCCL_IB_ADDR_RANGE'):
                    i += 2; continue
                expected.append(original[i]); i += 1
            self.assertEqual(stripped, expected)
            self.assertEqual(len(mounts), 7)
            self.assertTrue(all(m.endswith(',readonly') for m in mounts))
            baked = compose.baked_argv(prof, site, int(rank), image='test-image')
            self.assertEqual(baked.count('GLM53_RDMA_SWITCHED=1'), 1)
            self.assertTrue(any(prepare.native_name() in x for x in baked))
        self.assertEqual(prof, fixture()[0])

    def test_preparation_and_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'option'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.main(['--out', str(output)]), 0)
                self.assertEqual(prepare.main(['--out', str(output)]), 0)
            receipt = json.loads((output / 'switched-roce-receipt.json').read_text())
            self.assertIsNone(receipt['native_sha256'])
            self.assertFalse((output / receipt['native_file']).exists())
            (output / 'glm53_prefill_rdma/ring.py').write_text('different')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(prepare.main(['--out', str(output)]), 2)
            root = Path(tmp) / 'source'
            for rel in prepare.changes():
                target = root / 'src/tp4' / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / 'src/tp4' / rel).read_bytes())
            (root / 'src/tp4/glm53_prefill_rdma/_prb.cu').write_text('different')
            with self.assertRaisesRegex(ValueError, 'pin differs'):
                prepare.derive(root)

    @unittest.skipUnless(shutil.which('cc'), 'C compiler unavailable')
    def test_actual_c_cuda_selectors_and_python_mapping(self):
        files = prepare.derive()
        lean = files['glm53_lean_allreduce/_roce_proxy.c'].decode()
        cuda = files['glm53_prefill_rdma/_prb.cu'].decode()
        select = c_function(lean, 'switched_ipv4_gid').replace('\r\n', '\n')
        self.assertEqual(select, c_function(cuda, 'switched_ipv4_gid'))
        self.assertIn('grh.sgid_index = (uint8_t)hca->sgid', lean)
        self.assertIn('grh.sgid_index = (uint8_t)c->sgid[h]', cuda)
        self.assertIn('hca->sgid = c->gid_index', lean)  # unchanged positive-index path
        self.assertIn('required_hcas = (HCA_NAMES[1], HCA_NAMES[3])',
                      files['glm53_prefill_rdma/ring.py'].decode())
        source = r'''
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IBV_GID_TYPE_ROCE_V2 2
union ibv_gid { uint8_t raw[16]; };
struct ibv_gid_entry { union ibv_gid gid; int gid_type; };
struct ibv_context { struct ibv_gid_entry rows[260]; int query_error; };
static int ibv_query_gid_ex(struct ibv_context *c, uint32_t port, uint32_t i,
                            struct ibv_gid_entry *out, uint32_t flags) {
    (void)port; (void)flags;
    if (c->query_error) return c->query_error;
    *out = c->rows[i]; return 0;
}
''' + select + c_function(cuda, 'prb_switched') + c_function(cuda, 'direct_path') + r'''
static void ipv4(struct ibv_context *c, int i) {
    c->rows[i].gid_type = IBV_GID_TYPE_ROCE_V2;
    c->rows[i].gid.raw[10] = c->rows[i].gid.raw[11] = 255;
    c->rows[i].gid.raw[12] = 192; c->rows[i].gid.raw[13] = 0;
    c->rows[i].gid.raw[14] = 2; c->rows[i].gid.raw[15] = 1;
}
int main(void) {
    struct ibv_context a, b; char error[256] = {0};
    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    ipv4(&a, 5); ipv4(&b, 3);
    assert(switched_ipv4_gid(&a, 1, 6, "rail0", error, sizeof(error)) == 5);
    assert(switched_ipv4_gid(&b, 1, 6, "rail1", error, sizeof(error)) == 3);
    ipv4(&a, 2);
    assert(switched_ipv4_gid(&a, 1, 6, "rail0", error, sizeof(error)) < 0);
    assert(strstr(error, "ambiguous"));
    memset(&a, 0, sizeof(a)); a.rows[1].gid_type = 2; a.rows[1].gid.raw[0] = 254;
    ipv4(&a, 2); a.rows[2].gid_type = 1;
    assert(switched_ipv4_gid(&a, 1, 6, "rail0", error, sizeof(error)) < 0);
    assert(strstr(error, "no usable"));
    memset(&a, 0, sizeof(a)); ipv4(&a, 256);
    assert(switched_ipv4_gid(&a, 1, 257, "rail0", error, sizeof(error)) < 0);
    assert(strstr(error, "unrepresentable"));
    a.query_error = EIO;
    assert(switched_ipv4_gid(&a, 1, 6, "rail0", error, sizeof(error)) < 0);
    assert(strstr(error, "cannot inspect"));
    for (int mode = 0; mode < 2; ++mode) {
        setenv("GLM53_RDMA_SWITCHED", mode ? "1" : "0", 1);
        for (int rank = 0; rank < 4; ++rank)
            for (int peer = 0; peer < 4; ++peer)
                if ((rank - peer + 4) % 4 == 1 || (rank - peer + 4) % 4 == 3)
                    for (int lane = 0; lane < 2; ++lane)
                        printf("%d %d %d %d %d\n", mode, rank, peer, lane, direct_path(rank, peer, lane));
    }
    assert(direct_path(0, 2, 0) == -1 && direct_path(0, 1, 2) == -1);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / 'selectors.c'; binary = Path(tmp) / 'selectors'
            c.write_text(source)
            for language, standard in [('c', 'c11'), ('c++', 'c++17')]:
                compiled = subprocess.run(['cc', '-x', language, '-std=' + standard, '-Wall', '-Wextra', '-Werror',
                                           str(c), '-o', str(binary)],
                                          capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                output = subprocess.check_output([str(binary)], text=True)
        prefill = protocol(files['glm53_prefill_rdma/protocol.py'], 'cpu_native_parity')
        for line in output.splitlines():
            mode, rank, peer, lane, got = map(int, line.split())
            with patch.dict(os.environ, {'GLM53_RDMA_SWITCHED': str(mode)}):
                self.assertEqual(prefill.direct_path(rank, peer, lane), got)


if __name__ == '__main__':
    unittest.main()
