# Energy tutorials

In lesson 01, predict where a six-node spring chain settles and check the
prediction with litepgo. Choose [Python](01_spring_chain_python.ipynb) or
[C++20](01_spring_chain_cpp.ipynb). Both notebooks use the same derivation,
experiment, numerical checks, and Try it.

## Set up the C++ kernel

On macOS or Linux, install uv, micromamba, CMake 3.28 or newer, Ninja, and a
C++20 compiler. See the [build guide](../../docs/guide/build.md) for compiler
setup. On Windows, run the setup in WSL and open that checkout in a VS Code
WSL window.

From the repository root:

```sh
uv run scripts/setup_cpp_tutorial.py
```

This installs xeus-cpp in `build/tutorial-xcpp/`, builds the native library in
`build/cpp-tutorials/`, and registers **C++20 (litepgo)** for your user account.
Rerun the command after moving the checkout or changing native code. CI runs
the same setup and executes Energy 01 on macOS and Linux.

## Set up the Python kernel

```sh
uv sync --locked --group tutorials
cmake --preset release
cmake --build --preset release --target litepgo_python
```

## Run in VS Code

If VS Code was already open during C++ setup, run **Developer: Reload Window**
from the Command Palette so the Jupyter extension discovers the new kernel.
In the notebook's upper right, click the current kernel name or **Select Kernel**,
then choose **Select Another Kernel**. For the C++ notebook, choose
**Jupyter Kernel → C++20 (litepgo)**. For the Python notebook, choose
**Python Environments** and select the repository's `.venv/bin/python`.
Then choose **Run All**.

## Execute from fresh kernels

After setting up both kernels, run the notebooks from the repository root.

```sh
(
    set -e
    for language in python cpp; do
        uv run --no-sync jupyter nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=90 --output-dir="$PWD/build/energy-tutorials" \
            "tutorials/energy/01_spring_chain_${language}.ipynb"
    done
)
```

Each invocation starts its own kernel. Executed copies are stored under
`build/energy-tutorials/`. Keep source notebooks free of outputs and execution
counts.
