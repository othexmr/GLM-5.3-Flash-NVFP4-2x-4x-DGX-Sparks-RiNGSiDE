# SPDX-License-Identifier: Apache-2.0
"""Topology contracts for four/eight-node dual-root rings. No host access."""
import hashlib
import ipaddress
import json

FUNCTIONS = ('rocep1s0f0', 'rocep1s0f1', 'roceP2p1s0f0', 'roceP2p1s0f1')


def digest(topology):
    return hashlib.sha256(json.dumps(topology, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(topology):
    nodes, links = topology['nodes'], topology['links']
    n = len(nodes)
    if (n not in (4, 8) or any(type(node['rank']) is not int for node in nodes)
            or sorted(node['rank'] for node in nodes) != list(range(n))):
        raise ValueError('exactly four or eight unique ranks 0..N-1 required')
    management = [str(ipaddress.ip_address(node['management_address'])) for node in nodes]
    if len(set(management)) != n:
        raise ValueError('duplicate management address')
    if tuple(topology['functions']) != FUNCTIONS or len(links) != n:
        raise ValueError('four declared DGX functions and N cables required')
    adjacency = {r: set() for r in range(n)}
    used = {r: set() for r in range(n)}
    networks, addresses, edges = [], set(), set()
    for link in links:
        pair = link['ranks']
        if len(pair) != 2 or any(type(r) is not int or r not in adjacency for r in pair) or pair[0] == pair[1]:
            raise ValueError('invalid cable ranks')
        edge = tuple(sorted(pair))
        if edge in edges:
            raise ValueError('duplicate cable')
        edges.add(edge)
        net = ipaddress.ip_network(link['subnet'], strict=True)
        if net.version != 4 or net.prefixlen != 24 or any(net.overlaps(x) for x in networks):
            raise ValueError('cables require distinct IPv4 /24 networks')
        networks.append(net)
        port = link['port']
        if port not in ('f0', 'f1'):
            raise ValueError('invalid cable port')
        expected = [f'rocep1s0{port}', f'roceP2p1s0{port}']
        if link['lanes'] != expected:
            raise ValueError('cable must use one function on each PCI root')
        if topology.get('schema') == 3 and set(link.get('addresses', {})) != {str(r) for r in pair}:
            raise ValueError('cable addresses must name exactly its two ranks')
        for r, other in (pair, pair[::-1]):
            if port in used[r]:
                raise ValueError('physical port used twice')
            used[r].add(port)
            adjacency[r].add(other)
            host = link.get('addresses', {}).get(str(r))
            if topology.get('schema') == 3 and (not isinstance(host, dict) or set(host) != set(expected)):
                raise ValueError('every cable endpoint needs both function addresses')
            if host is not None:
                for value in host.values():
                    address = ipaddress.ip_address(value)
                    if address not in net or address in (net.network_address, net.broadcast_address) or address in addresses:
                        raise ValueError('duplicate or invalid fabric host address')
                    addresses.add(address)
    if any(len(v) != 2 for v in adjacency.values()):
        raise ValueError('every rank must have two different neighbors')
    reached, frontier = set(), [0]
    while frontier:
        r = frontier.pop()
        if r not in reached:
            reached.add(r)
            frontier.extend(adjacency[r] - reached)
    if len(reached) != n:
        raise ValueError('cabling is disconnected; one cycle required')
    return adjacency


def generate(ranks, subnet_base='10.100.224.0/24'):
    if ranks not in (4, 8):
        raise ValueError('supported ring sizes: 4, 8')
    first = ipaddress.ip_network(subnet_base, strict=True)
    if first.version != 4 or first.prefixlen != 24:
        raise ValueError('subnet base must be an IPv4 /24 network')
    links = []
    for r in range(ranks):
        network = ipaddress.IPv4Network((int(first.network_address) + 256 * r, 24))
        port = 'f1' if r % 2 == 0 else 'f0'
        lanes = [f'rocep1s0{port}', f'roceP2p1s0{port}']
        pair = [r, (r + 1) % ranks]
        links.append({'ranks': pair, 'port': port, 'lanes': lanes, 'subnet': str(network),
                      'addresses': {str(k): {hca: str(network.network_address + 1 + k + 32 * root)
                                            for root, hca in enumerate(lanes)} for k in pair}})
    topology = {'schema': 3, 'topology': f'tp{ranks}-ring', 'functions': list(FUNCTIONS),
                'nodes': [{'rank': r, 'management_address': f'192.0.2.{10+r}'} for r in range(ranks)],
                'links': links}
    validate(topology)
    return topology


def cable_subnets(topology):
    validate(topology)
    return ','.join(link['subnet'] for link in topology['links'])
