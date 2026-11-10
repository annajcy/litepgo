# Custom energies and responses

Implement a scalar $E:\mathbb R^n\to\mathbb R$ or a vector $r:\mathbb R^n\to\mathbb R^m$
through fixed/dynamic-pattern subclasses. Read [evaluation contracts](evaluation.md)
and the module's [error and feasibility rules](index.md#errors-and-feasibility)
before overriding the numerical hooks.

## Implement a scalar energy

Derive from `FixedPatternEnergy(pattern)` or
`DynamicPatternEnergy(num_dofs)`. The latter additionally
implements `hessian_pattern_at(x)`. In Python, override

```python
def eval_values(self, x, need_gradient, need_hessian):
    # Exact outputs: (scalar, gradient_or_None, hessian_or_None).
    ...

# Optional PSD fast path; same arguments and return shape.
def eval_projected_values(self, x, need_gradient, need_hessian):
    ...
```

Omit `eval_projected_values` to use automatic spectral projection. Return `None`
for an infeasible state. Exact Hessians must fit the declared structure;
projected Hessians may have their own structure. Optional `feasible_step(x, dx)`
defaults to one and must bound the complete closed segment.

C++ fixed/dynamic subclasses override
`double eval_values(x, Vec* gradient, SpMat* hessian) const`. Null pointers mean
unrequested outputs. Overwrite all requested entries, including structural
zeros. Dynamic subclasses also implement `SpMat hessian_pattern_at(x) const`.
The [custom-energy example](../../examples/energy/custom-energy.md) demonstrates
both languages.

C++ implementations that produce their own matrices may derive directly from
`Energy(num_dofs)` and override `eval_impl(x, need_gradient, need_hessian) const`,
returning `EnergyValues`. The optional `eval_projected_values(x, exact_evaluation)`
returns value, gradient and a PSD Hessian; its default reuses the exact evaluation
and projects its Hessian. More specialized implementations can override
`make_evaluation(x, borrow_input)` to retain child states. `MeshEnergy` retains its
batch hooks and local projection path. All outputs pass through the evaluation's validation.

For a custom C++ evaluation, forward the input policy to its base constructor:

```cpp
MyEvaluation::MyEvaluation(const MyEnergy& model, const ConstVecRef& x, bool borrow_input)
    : EnergyEvaluation(model, x, borrow_input) {}

std::shared_ptr<EnergyEvaluation> MyEnergy::make_evaluation(
    const ConstVecRef& x, bool borrow_input) const {
    return std::make_shared<MyEvaluation>(*this, x, borrow_input);
}
```

Public `evaluate(x)` passes `false`, so the state copies its input. Internal
weighted-sum child creation passes `true`, allowing a read-only borrow of the
parent's snapshot; taking a copy instead is also safe. Borrowed children must not
outlive their parent or escape to external callers. The protected `x_` member is
now a read-only `ConstVecRef`, backed by either owned storage or the parent.
Evaluation objects cannot be copied or moved; move their owning pointers instead.
This replaces the former one-argument C++ factory hook. The numerical hooks and
public C++/Python evaluation methods are unchanged.

## Implement a vector response

`FixedPatternResponse(pattern)` gets $m,n$ from the pattern's rows and columns.
`DynamicPatternResponse(num_inputs, num_outputs)` additionally implements
`jacobian_pattern_at(x)`. Override `eval_impl(x, need_jacobian)` and return the
value for a value-only call or `(value, J)` for a Jacobian request. Return `None`
for an infeasible point. Native `eval_impl(x, SpMat* J)` uses a nullable
Jacobian pointer and overwrites its prepared numerical entries.

```python
value = response.evaluate(x).value
evaluation = response.evaluate(x)
J = evaluation.jacobian()
value = evaluation.value
```

Ordinary calls prepare their required structure in local storage. No explicit
pattern preparation or output buffer protocol is exposed. See
[Response evaluation](responses.md) for all public methods, subclass hooks, and
migration from the previous API. The
[response example](../../examples/energy/custom-response.md) demonstrates fixed
and dynamic structure and a Gauss–Newton solve.

Optional second derivatives use an additional Hessian pattern and the
`eval_hessian_impl` hook. See [Response Hessians](response-hessians.md) for the
weighted definition and constructor signatures. First-order subclasses need no
Hessian hook, but must use the current evaluation hook names.
