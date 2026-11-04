from concurrent.futures import ThreadPoolExecutor

import pytest

import litepgo


active = litepgo._core._active_tbb_thread_limit


def test_nested_limits_restore_after_exception():
    original = active()
    with litepgo.tbb_thread_limit(2):
        outer = active()
        assert outer <= 2
        with litepgo.tbb_thread_limit(4):
            assert active() == outer
        with pytest.raises(RuntimeError, match="unwind"):
            with litepgo.tbb_thread_limit(1):
                assert active() == 1
                raise RuntimeError("unwind")
        assert active() == outer
    assert active() == original


def test_limit_is_shared_across_python_threads():
    original = active()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with litepgo.tbb_thread_limit(1):
            assert executor.submit(active).result() == 1
        assert executor.submit(active).result() == original


@pytest.mark.parametrize("value", [True, False, 1.5, "4", None])
def test_rejects_non_integer(value):
    with pytest.raises(TypeError):
        with litepgo.tbb_thread_limit(value):
            pytest.fail("invalid limit entered")


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive(value):
    with pytest.raises(ValueError):
        with litepgo.tbb_thread_limit(value):
            pytest.fail("invalid limit entered")


def test_context_creation_does_not_install_limit():
    original = active()
    context = litepgo.tbb_thread_limit(1)
    assert active() == original
    with context:
        assert active() == 1
    assert active() == original
