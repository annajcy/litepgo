from pathlib import Path
import subprocess
import sys
import pytest
from litepgo import _core


@pytest.mark.parametrize('folder,name',[('spatial','spatial_query'),('fem/embedding','point_embedding')])
def test_cpp_python_query_examples_match(folder,name):
    root = Path(__file__).resolve().parents[2]
    executable = Path(_core.__file__).parents[2]/'examples'/folder/name
    cpp = subprocess.check_output([str(executable)],text=True)
    python = subprocess.check_output([sys.executable,str(root/'examples'/folder/(name+'.py'))],text=True)
    assert cpp == python
