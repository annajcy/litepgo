import pytest

from litepgo import mesh


def test_write_abc_from_objs(tmp_path):
    first, second = (tmp_path / name for name in ("first.obj", "second.obj"))
    first.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    second.write_text("v 0 0 1\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    output = tmp_path / "motion.abc"
    if not mesh.has_alembic:
        with pytest.raises(RuntimeError, match="LITEPGO_WITH_ALEMBIC=ON"):
            mesh.write_abc_from_objs(output, [first, second], fps=24)
        return

    with pytest.raises(ValueError, match="one OBJ frame"):
        mesh.write_abc_from_objs(output, [], fps=24)
    for fps in (0, -24, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fps"):
            mesh.write_abc_from_objs(output, [first], fps=fps)
    assert not output.exists()
    mesh.write_abc_from_objs(output, [first, str(second)], fps=24)
    original = output.read_bytes()
    assert original.startswith(b"Ogawa")
    second.write_text("v 0 0 1\nv 1 0 0\nv 0 1 0\nf 1 3 2\n")
    with pytest.raises(RuntimeError, match=r"frame 1 .*second\.obj.*topology"):
        mesh.write_abc_from_objs(output, [first, second], fps=24)
    with pytest.raises(RuntimeError, match=r"frame 1 .*missing\.obj"):
        mesh.write_abc_from_objs(output, [first, tmp_path / "missing.obj"], fps=24)
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == {first, second, output}
    mesh.write_abc_from_objs(output, [first], fps=24)
    assert output.read_bytes() != original
