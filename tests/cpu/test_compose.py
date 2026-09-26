# SPDX-License-Identifier: Apache-2.0
"""launch/compose.py renders, per rank, a Compose service with exactly the docker-run semantics of the canonical launch
for the self-built image; launch/up.sh and launch/down.sh are dry runs unless --go and start workers before rank 0."""
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compose = module('switchless_compose', 'launch/compose.py')
render = module('switchless_render_test', 'launch/render.py')


def profile(topo):
    return json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())


def fixture(topo):
    return json.loads((ROOT / 'launch/profiles' / topo / 'canonical-launch.json').read_text())


def read_env(text):
    """A .env file as Compose reads the rendered one: KEY='value' lines, # comments."""
    env = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        assert sep and value[:1] == "'" and value[-1:] == "'", line
        env[key] = value[1:-1]
    return env


def read_compose(text):
    """compose.yaml is JSON after its comment lines (JSON is valid YAML)."""
    return json.loads('\n'.join(line for line in text.splitlines() if not line.startswith('#')))


def interpolate(value, env):
    if isinstance(value, str):
        return compose.resolve(value, env)
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    return value


def run_options(argv, image):
    """The docker-run argv as ordered option pairs, the image and the command."""
    assert argv[:2] == ['docker', 'run']
    end = argv.index(image)
    pairs, i = [], 2
    while i < end:
        if argv[i] == '-d':
            pairs.append(('-d', None))
            i += 1
        else:
            pairs.append((argv[i], argv[i + 1]))
            i += 2
    return pairs, argv[end], argv[end + 1:]


def service_options(svc):
    """The Compose service mapped back to docker-run option pairs, independently of launch/compose.py."""
    known = {'image', 'build', 'environment', 'volumes', 'container_name', 'network_mode', 'ipc', 'deploy', 'cap_add',
             'security_opt', 'devices', 'ulimits', 'shm_size', 'labels', 'command'}
    unknown = set(svc) - known
    assert not unknown, unknown
    out = {'-e': list(svc['environment']), '--mount': [], '--name': [svc['container_name']],
           '--network': [svc['network_mode']], '--ipc': [svc['ipc']], '--cap-add': svc.get('cap_add', []),
           '--security-opt': svc.get('security_opt', []), '--shm-size': [svc['shm_size']],
           '--label': svc.get('labels', []), '--device': [], '--ulimit': [], '--gpus': []}
    for v in svc['volumes']:
        assert v['type'] == 'bind' and v['bind'] == {'create_host_path': False} and set(v) == {
            'type', 'source', 'target', 'read_only', 'bind'}, v
        out['--mount'].append(f"type=bind,src={v['source']},dst={v['target']}" + (',readonly' if v['read_only'] else ''))
    for d in svc.get('devices', []):
        host, _, container = d.partition(':')
        assert host == container, d
        out['--device'].append(host)
    for name, lim in svc.get('ulimits', {}).items():
        out['--ulimit'].append(f"{name}={lim['soft']}:{lim['hard']}")
    assert svc['deploy'] == {'resources': {'reservations': {'devices': [
        {'driver': 'nvidia', 'count': 'all', 'capabilities': ['gpu']}]}}}
    out['--gpus'].append('all')
    return out


class ComposeProjects(unittest.TestCase):
    def check_rank(self, topo, site, rank):
        p = profile(topo)
        doc, env = compose.project(p, site, rank, compose.CONTEXT)
        text = compose.compose_text(doc, topo, rank)
        loaded = read_compose(text)
        self.assertEqual(loaded, doc)
        env = read_env(compose.env_text(topo, rank, env))
        svc = interpolate(loaded['services'][compose.SERVICE], env)
        image = compose.build_image(site, p)
        argv = compose.baked_argv(p, site, rank)
        pairs, argv_image, command = run_options(argv, image)
        # element by element: every option pair of the rendered argv has its Compose counterpart, in order where
        # docker keeps order (environment, mounts), and nothing else is in the service
        mapped = service_options(svc)
        by_flag = {}
        for flag, value in pairs:
            by_flag.setdefault(flag, []).append(value)
        self.assertEqual(by_flag.pop('-d'), [None])  # launch/up.sh runs `docker compose up -d`
        self.assertEqual(set(by_flag), {k for k, v in mapped.items() if v}, (topo, rank))
        for flag, values in by_flag.items():
            self.assertEqual(values, mapped[flag], (topo, rank, flag))
        self.assertEqual(svc['image'], argv_image)
        self.assertEqual(svc['command'], command)
        self.assertEqual(svc['build']['target'], f'profile-{topo}')
        self.assertEqual(svc['build']['dockerfile'], 'docker/Dockerfile')
        return argv

    def test_every_rank_of_the_canonical_site_is_equivalent(self):
        for topo in ('tp4', 'tp2'):
            site = fixture(topo)['site']
            for rank in range(profile(topo)['ranks']):
                self.check_rank(topo, site, rank)

    def test_baked_argv_is_the_canonical_launch_without_the_overlay_mounts(self):
        for topo in ('tp4', 'tp2'):
            p, fx = profile(topo), fixture(topo)
            served = {f['target'] for f in json.loads((ROOT / 'sources/installed-files.json').read_text())
                      ['profiles'][topo]['files']}
            overlay = fx['site']['SWITCHLESS_OVERLAY_ROOT']
            for rank in range(p['ranks']):
                canonical = fx['argv'][str(rank)]
                image = compose.build_image(fx['site'], p)
                expected, dropped, i = [], set(), 0
                while i < len(canonical):
                    if canonical[i] == '--mount' and canonical[i + 1].startswith(f'type=bind,src={overlay}/'):
                        kv = dict(x.split('=', 1) for x in canonical[i + 1].split(',')[:-1])
                        self.assertEqual(kv['src'], overlay + kv['dst'])
                        self.assertTrue(canonical[i + 1].endswith(',readonly'))
                        dropped.add(kv['dst'])
                        i += 2
                        continue
                    expected.append(image if canonical[i] == p['image']['id'] else canonical[i])
                    i += 1
                self.assertEqual(compose.baked_argv(p, fx['site'], rank), expected, (topo, rank))
                # the image bakes exactly the files the dropped mounts served (docker/tools/bake_overlay.py)
                self.assertEqual(dropped, served, (topo, rank))

    def test_site_values_are_checked_like_render(self):
        site = dict(fixture('tp4')['site'])
        site['SWITCHLESS_MODEL_DIR'] = '/srv/a,readonly=false'
        with self.assertRaisesRegex(ValueError, 'must not contain a comma'):
            compose.project(profile('tp4'), site, 0, compose.CONTEXT)
        site = dict(fixture('tp4')['site'])
        site['SWITCHLESS_HF_CACHE_DIR'] = "/srv/it's"
        with self.assertRaisesRegex(ValueError, 'quote'):
            _, env = compose.project(profile('tp4'), site, 0, compose.CONTEXT)
            compose.env_text('tp4', 0, env)

    def test_literal_dollars_are_escaped(self):
        self.assertEqual(compose.compose_token('a$b${SWITCHLESS_API_PORT}${RANK}'),
                         'a$$b${SWITCHLESS_API_PORT}${SWITCHLESS_RANK}')
        self.assertEqual(compose.resolve('a$$b${X}', {'X': '1'}), 'a$b1')

    def test_source_service_is_pinned(self):
        build = compose.SOURCE_BUILD
        self.assertRegex(build['context'], r'^https://github\.com/[\w.-]+/vllm\.git#[0-9a-f]{40}$')
        for key in ('BUILD_BASE_IMAGE', 'FINAL_BASE_IMAGE'):
            self.assertRegex(build['args'][key], r'@sha256:[0-9a-f]{64}$')

    def test_cli_writes_one_project_per_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, '-B', str(ROOT / 'launch/compose.py'), '--profile', 'tp4', '--site',
                                str(ROOT / 'site.env.example'), '--out', tmp], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for rank in range(4):
                d = Path(tmp) / f'rank{rank}'
                doc = read_compose((d / 'compose.yaml').read_text())
                env = read_env((d / '.env').read_text())
                self.assertEqual(env['SWITCHLESS_RANK'], str(rank))
                svc = interpolate(doc['services']['glm53'], env)
                self.assertEqual('--headless' in svc['command'], rank != 0)
                self.assertEqual(doc['services']['vllm-source']['profiles'], ['build'])


def fake_path(tmp):
    """ssh, scp and docker that only record a call, so a dry run can prove it called none of them."""
    log = Path(tmp) / 'calls.log'
    bindir = Path(tmp) / 'bin'
    bindir.mkdir()
    for name in ('ssh', 'scp', 'docker'):
        f = bindir / name
        f.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 1\n')
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    return f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", log


class Launchers(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in ('launch/up.sh', 'launch/down.sh'):
            r = subprocess.run(['bash', '-n', str(ROOT / rel)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            text = (ROOT / rel).read_text()
            self.assertTrue(text.startswith('#!/usr/bin/env bash\n'), rel)
            self.assertIn('set -euo pipefail', text, rel)

    def dry_run(self, script, *args, site_extra=''):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / 'site.env'
            site.write_text((ROOT / 'site.env.example').read_text() + site_extra)
            path, log = fake_path(tmp)
            r = subprocess.run(['bash', str(ROOT / script), '--site', str(site), *args], capture_output=True,
                               text=True, env=dict(os.environ, PATH=path))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertFalse(log.exists(), 'a dry run called: ' + (log.read_text() if log.exists() else ''))
            return r.stdout

    def test_up_dry_run_starts_workers_before_rank_0(self):
        out = self.dry_run('launch/up.sh', '--profile', 'tp4', '--build')
        self.assertIn('Dry run', out)
        self.assertEqual(re.findall(r'^# Rank (\d) on', out, re.M), ['3', '2', '1', '0'])
        starts = [m.start() for m in re.finditer('docker compose up -d --no-build --pull never glm53', out)]
        self.assertEqual(len(starts), 4)
        self.assertLess(out.index('docker compose --profile build build'), starts[0])
        self.assertEqual(out.count('docker load'), 3)
        self.assertIn('/health', out)
        self.assertIn('/v1/models', out)
        self.assertRegex(out, r'# Endpoint: http://192\.0\.2\.10:8888/v1 ')

    def test_up_without_build_builds_nothing(self):
        out = self.dry_run('launch/up.sh', '--profile', 'tp2', site_extra=(
            'SWITCHLESS_RANK_HOST_IPS=192.0.2.20,192.0.2.21\nSWITCHLESS_MASTER_ADDR=192.0.2.20\n'))
        self.assertNotIn(' build', out.replace('--no-build', ''))
        self.assertNotIn('docker load', out)
        self.assertEqual(re.findall(r'^# Rank (\d) on', out, re.M), ['1', '0'])

    def test_down_dry_run_stops_rank_0_first(self):
        out = self.dry_run('launch/down.sh', '--profile', 'tp4')
        self.assertEqual(re.findall(r'deploy/tp4/rank(\d) && docker compose down', out), ['0', '1', '2', '3'])

    def test_unknown_argument_is_refused(self):
        r = subprocess.run(['bash', str(ROOT / 'launch/up.sh'), '--profile', 'tp4', '--site',
                            str(ROOT / 'site.env.example'), '--pull'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == '__main__':
    unittest.main()
