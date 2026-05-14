import pytest
import glob
import os
import shutil
import sys

sys.path.append('./src')
sys.path.append('./src/lib')


@pytest.fixture(autouse=True)
def _cleanup_pycaches():
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
