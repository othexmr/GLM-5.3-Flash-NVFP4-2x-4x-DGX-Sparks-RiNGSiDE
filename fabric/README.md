# Fabric

## TP4: four DGX Sparks in a ring

Each Spark has one ConnectX-7 with two ports, and each port exposes two PCIe functions (one per PCIe root):
`rocep1s0f0`/`roceP2p1s0f0` on one port and `rocep1s0f1`/`roceP2p1s0f1` on the other. The four Sparks form a cycle
with one cable between neighbours; there is no switch.

| Cable | Ports used (both ends) | Subnet |
|---|---|---|
| rank 0 - rank 1 | `f1` functions | 10.100.224.0/24 |
| rank 1 - rank 2 | `f0` functions | 10.100.225.0/24 |
| rank 2 - rank 3 | `f1` functions | 10.100.226.0/24 |
| rank 3 - rank 0 | `f0` functions | 10.100.227.0/24 |

Two parts of the TP4 profile depend on this plan:

- The NCCL build (`patches/nccl/0002`) validates listener addresses and accepts only IPv4 addresses in
  10.100.224.0/24 to 10.100.227.0/24, one subnet per cable, with both PCIe functions of each cable's port addressed
  (four IPv4 GIDs per node). Another address plan needs a source change and a new build.
- The lean all-reduce (`src/tp4/glm53_lean_allreduce/protocol.py`, `DIRECT_PATHS`) connects each rank only to its two
  neighbours and uses the port of the table above for each cable, with the two PCIe functions of that port as two
  lanes. Another cabling needs that table changed.

NCCL runs Ring only (`NCCL_ALGO=Ring`); the switchless patch skips Tree and PAT connections, which a ring cannot
serve. The management network (any Ethernet) carries the bootstrap traffic (`SWITCHLESS_SOCKET_IFNAME`).

### Fixed device names and GID index

The two RDMA transports of the TP4 profile do not take their devices from the site file:

- the lean all-reduce (`src/tp4/glm53_lean_allreduce/lean_tp4.py`, `HCA_NAMES`) and the prefill RDMA ring
  (`src/tp4/glm53_prefill_rdma/protocol.py`, `HCA_NAMES`; `ring.py`) open exactly `rocep1s0f0`, `rocep1s0f1`,
  `roceP2p1s0f0` and `roceP2p1s0f1`, and refuse to start (the rank exits) when one is missing;
- both use GID index `NCCL_IB_GID_INDEX`, default 3, on every function. The TP4 profile does not set that variable,
  so index 3 must be the RoCE v2 entry of the function's IPv4 address. On a port with one IPv4 address and no other
  addresses this is the usual layout (0/1: link-local RoCE v1/v2, 2/3: IPv4 RoCE v1/v2); extra addresses on the
  interface can move it.

`SWITCHLESS_IB_HCA` configures NCCL only. Nodes whose functions have other names, or whose RoCE v2 IPv4 GID is at
another index, need source changes (or `NCCL_IB_GID_INDEX` added to the launch, which is then a new plan).

`preflight_tp4.py` checks this on a node before the first launch (read-only, sysfs only):

```sh
python3 fabric/preflight_tp4.py --rank N      # N = this node's rank in the table above; exit 0 = ready
```

It checks that the four functions exist with an ACTIVE Ethernet port, that GID index 3 is a RoCE v2 entry holding an
IPv4-mapped address, and that the address lies in the subnet of the function's cable.

## TP2: two DGX Sparks

The measured TP2 profile uses one cable between the two Sparks on the `f1` port, both PCIe functions
(`rocep1s0f1,roceP2p1s0f1`), stock NCCL, and the fabric link itself for the bootstrap traffic (the rank addresses
and `SWITCHLESS_SOCKET_IFNAME` are on that link).

## Checks before a launch

- TP4: `preflight_tp4.py --rank N` exits 0 on every node, and every node reaches both neighbours on both lanes.
- Link speed on every port (the lab verified 200 Gb/s on all eight ports).
- `topology.example.json` describes the TP4 ring in machine-readable form (the preflight reads its cable subnets);
  replace the example management addresses.
