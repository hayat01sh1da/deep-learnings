import pytest
import re
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def __cleanup_caches__():
    yield
    cache_dir = re.compile(r'^(?:__pycache__|\.pytest_cache|\.mypy_cache)$')
    for root, dirs, _ in os.walk('.'):
        for name in list(dirs):
            if cache_dir.match(name):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                dirs.remove(name)
