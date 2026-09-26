# Experimental TP8 NCCL

Apply the series to NVIDIA NCCL at the commit in `sources.lock.json`.
The first two patches are unchanged copies of the release series, with its
SparkRing and switchless-nccl attribution. The third patch admits a configured
list of four or eight distinct cable networks; each listener still advertises
four GIDs on two cables and two PCI roots. Ring-only routing remains required.
Set `NCCL_SWITCHLESS_CABLE_SUBNETS` to the topology's comma-separated IPv4 /24s.
Missing, malformed or duplicate networks fail closed.

This component has a separate source tree, build stage and installation path.
The released TP4 library is untouched. A successful compilation establishes no
multi-rank correctness or performance result. See `docs/tp8.md` for the gates.
