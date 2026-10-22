import litepgo


def test_native_module_loads():
    assert litepgo._core.__doc__ == "litepgo native extension"
