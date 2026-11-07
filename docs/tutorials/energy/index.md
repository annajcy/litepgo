# Energy tutorial

Use a six-node spring chain to learn how litepgo represents energy and
response. Each lesson runs independently in either Python or C++20. Both
versions use the same inputs, derivation, checks, and Try it.

| Lesson | What you do | Python | C++20 |
| --- | --- | --- | --- |
| 1. Spring-chain equilibrium | Use a quadratic energy and Newton to predict where the chain settles | [Read](./01_spring_chain_python.md) | [Read](./01_spring_chain_cpp.md) |
| 2. Spring differences and gradients | Use an affine response, JVP, and VJP to recover the energy gradient | [Read](./02_spring_response_python.md) | [Read](./02_spring_response_cpp.md) |

Each lesson page shows output from a fresh kernel and links to the source
notebook. To run a downloaded notebook, open it in the litepgo checkout and
set up the corresponding kernel.

For Python, run the following from the repository root and select
`.venv/bin/python` in your notebook editor.

```sh
uv sync --locked --group tutorials
cmake --preset release
cmake --build --preset release --target litepgo_python
```

For C++20, install micromamba alongside the [native build tools](../../guide/build.md).
Run the setup command on macOS or Linux. On Windows, use WSL.

```sh
uv run scripts/setup_cpp_tutorial.py
```

If VS Code was already open, run **Developer: Reload Window**. Open the C++
notebook, choose **Select Another Kernel → Jupyter Kernel → C++20 (litepgo)**,
then choose **Run All**.
