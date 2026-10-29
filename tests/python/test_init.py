import sys

import litepgo
import litepgo.linalg.sparse_solver as la


def test_native_module_loads():
    assert litepgo._core.__doc__ == "litepgo native extension"


def test_build_info_reports_dependencies():
    info = litepgo.build_info()
    assert info["version"] == "0.0.1"
    assert info["eigen_version"] == "5.0.1"
    assert info["tbb_runtime_version"]
    assert info["sparse_backends"][0] == "EigenLDLT"
    assert ("AccelerateLDLT" in info["sparse_backends"]) == (sys.platform == "darwin")
    assert ("PardisoLDLT" in info["sparse_backends"]) == hasattr(la, "PardisoLDLT")
