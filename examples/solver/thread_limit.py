"""Run the Newton chain example with scoped TBB limits.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/solver/thread_limit.py
"""

from litepgo import tbb_thread_limit
from newton_chain import main as solve_chain


class ExampleError(Exception):
    pass


def main():
    with tbb_thread_limit(4):
        print("Outer TBB limit: at most 4")
        solve_chain()
        try:
            with tbb_thread_limit(1):
                print("Nested TBB limit: 1")
                solve_chain()
                raise ExampleError("demonstrate exception cleanup")
        except ExampleError:
            pass
        print("After exception: outer TBB limit applies again")
        solve_chain()
    print("After outer scope: previous TBB policy restored")


if __name__ == "__main__":
    main()
