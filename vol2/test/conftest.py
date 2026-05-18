import pytest
import re
import os
import shutil
import sys

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(_SRC)
sys.path.append(os.path.join(_SRC, 'concerns'))
sys.path.append(os.path.join(_SRC, 'layers'))
sys.path.append(os.path.join(_SRC, 'models'))
sys.path.append(os.path.join(_SRC, 'optimisers'))


@pytest.fixture(autouse=True)
def __cleanup_caches__():
    yield
    cache_dir = re.compile(r'^(?:__pycache__|\.pytest_cache|\.mypy_cache)$')
    for root, dirs, _ in os.walk('.'):
        for name in list(dirs):
            if cache_dir.match(name):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                dirs.remove(name)
