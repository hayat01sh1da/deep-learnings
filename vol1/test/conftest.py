import pytest
import glob
import os
import shutil
import sys

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(_SRC)
sys.path.append(os.path.join(_SRC, 'lib'))


@pytest.fixture(autouse=True)
def __cleanup_caches__():
    before = set(
        glob.glob(
            os.path.join(
                '.',
                '**',
                '__pycache__'),
            recursive=True))
    yield
    for pycache in before:
        if os.path.exists(pycache):
            shutil.rmtree(pycache)
