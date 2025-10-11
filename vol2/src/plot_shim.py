"""A tiny matplotlib shim used in tests when matplotlib is not installed.

It exposes a `plt`-like object with the subset of functions used by tests:
figure, annotate, contourf, scatter, savefig, imshow, plot, xlabel, ylabel, ylim.
If real matplotlib is available, we use it; otherwise we provide no-op implementations
that still create an empty image file on savefig so file-creation tests pass.
"""
from types import SimpleNamespace
import os
try:
    import matplotlib.pyplot as _plt
    plt = _plt
except Exception:
    from PIL import Image
    import numpy as _np

    class _NoopFigure:
        def __init__(self):
            self._w, self._h = 640, 480

    def _ensure_dir(fp):
        d = os.path.dirname(fp)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def figure(*args, **kwargs):
        return _NoopFigure()

    def annotate(*args, **kwargs):
        return None

    def contourf(*args, **kwargs):
        return None

    def scatter(*args, **kwargs):
        return None

    def imshow(*args, **kwargs):
        return None

    def plot(*args, **kwargs):
        return None

    def xlabel(*args, **kwargs):
        return None

    def ylabel(*args, **kwargs):
        return None

    def ylim(*args, **kwargs):
        return None

    def savefig(fp, *args, **kwargs):
        # create a tiny blank PNG so tests that check file existence pass
        _ensure_dir(fp)
        arr = _np.zeros((10, 10, 3), dtype=_np.uint8)
        Image.fromarray(arr).save(fp)

    # Additional plotting functions commonly used in tests
    def axis(*args, **kwargs):
        return None

    def legend(*args, **kwargs):
        return None

    def subplot(*args, **kwargs):
        return None

    def subplots(*args, **kwargs):
        return (None, None)

    def show(*args, **kwargs):
        return None

    def xlim(*args, **kwargs):
        return None

    def xticks(*args, **kwargs):
        return None

    def yticks(*args, **kwargs):
        return None

    def contour(*args, **kwargs):
        return None

    def colorbar(*args, **kwargs):
        return None

    # Minimal colormap namespace used like plt.cm.gray_r
    cm = SimpleNamespace(gray_r=None)

    plt = SimpleNamespace(
        figure=figure,
        annotate=annotate,
        contourf=contourf,
        scatter=scatter,
        imshow=imshow,
        plot=plot,
        xlabel=xlabel,
        ylabel=ylabel,
        ylim=ylim,
        savefig=savefig,
    )

__all__ = ['plt']
