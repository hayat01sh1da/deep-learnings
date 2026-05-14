import glob
import os
import shutil
import sys

sys.path.append('./2_natural_coding/src')


import pytest


@pytest.fixture(autouse=True)
def _cleanup_pycaches():
    before = set(glob.glob(os.path.join('.', '**', '__pycache__'), recursive=True))
    yield
    for pycache in before:
        if os.path.exists(pycache):
            shutil.rmtree(pycache)
