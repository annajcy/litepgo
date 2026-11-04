# Scoped TBB thread limits

Both Python and C++ examples demonstrate nested limits and exception cleanup.

This Python example reuses the [Newton chain](newton-chain.md) problem and its
solution checks. It solves under an outer limit of four, then under a nested
limit of one, and again after an exception leaves the inner scope.

```sh
cmake --build build/release --target litepgo_python
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/thread_limit.py
```

The inner limit is removed even on an exception. The outer limit remains until
its own scope exits, at which point the previous TBB policy applies again.
Other active limits can reduce the allowed parallelism further: overlapping
limits take the minimum, rather than the most recently entered value.

The printed labels describe the requested policy, not measured thread counts.
This small problem checks correctness and API usage; it is not a scaling benchmark.
The limit affects the shared TBB runtime across Python threads, including
litepgo's MKL/PARDISO backend. It does not control Accelerate or independent
NumPy/SciPy thread pools. See [threading semantics](../../guide/common.md#limiting-tbb-parallelism).

::: details Complete Python source — `thread_limit.py`

<<< ../../../examples/solver/thread_limit.py

:::

## C++ runtime controls

The C++ version uses TBB directly and checks a parallel loop's results under
each scope. It also demonstrates `task_arena(2).execute(...)` for arena-local
execution. Its printed values are active global limits, not measured worker
counts. Checks remain enabled in Release builds.

```sh
cmake --build build/release --target thread_limit
build/release/examples/solver/thread_limit
ctest --test-dir build/release -R '^TbbThreadLimitExample$' --output-on-failure
```

::: details Complete C++ source — `thread_limit.cpp`

<<< ../../../examples/solver/thread_limit.cpp

:::
