"""Distributed allocation benchmark — agents coordinate to solve joint tasks."""

from swarm.benchmarks.coordination.distributed_allocation import (
    AllocationInstance,
    DistributedAllocationBenchmark,
)

__all__ = ["DistributedAllocationBenchmark", "AllocationInstance"]
