// Build: cmake --build build/release --target thread_limit
// Run: build/release/examples/solver/thread_limit
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Control = tbb::global_control;

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("Thread limit example check failed");
}

void compute() {
    std::vector<int> values(10000);
    tbb::parallel_for(size_t(0), values.size(), [&](size_t i) { values[i] = static_cast<int>(i) * 2; });
    for (size_t i = 0; i < values.size(); ++i)
        check(values[i] == static_cast<int>(i) * 2);
}

struct ExampleError {};
} // namespace

int main() {
    const auto active = [] { return Control::active_value(Control::max_allowed_parallelism); };
    const auto original = active();
    {
        Control outer(Control::max_allowed_parallelism, 4);
        const auto outer_limit = active();
        check(outer_limit <= 4);
        std::cout << "Outer TBB limit: " << outer_limit << '\n';
        compute();
        try {
            Control inner(Control::max_allowed_parallelism, 1);
            check(active() == 1);
            compute();
            throw ExampleError{};
        } catch (const ExampleError&) {
        }
        check(active() == outer_limit);
        std::cout << "After exception: " << active() << '\n';
        // Arena-local execution is available directly in C++.
        tbb::task_arena arena(2);
        arena.execute([] {
            check(tbb::this_task_arena::max_concurrency() == 2);
            compute();
        });
    }
    check(active() == original);
    std::cout << "Restored TBB limit: " << active() << '\n';
}
