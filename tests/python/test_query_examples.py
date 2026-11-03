from pathlib import Path
import subprocess
import sys
import pytest
from litepgo import _core


@pytest.mark.parametrize('folder,name',[('spatial','spatial_query'),('fem/embedding','point_embedding')])
def test_cpp_python_query_examples_match(folder,name):
    root = Path(__file__).resolve().parents[2]
    executable = Path(_core.__file__).parents[2]/'examples'/folder/name
    if sys.platform == 'win32':
        executable = executable.with_suffix('.exe')
    python = subprocess.check_output([sys.executable,str(root/'examples'/folder/(name+'.py'))],text=True,timeout=60)
    assert python
    cpp = subprocess.check_output([str(executable)],text=True,timeout=60)
    assert cpp == python
