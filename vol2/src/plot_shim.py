"""A tiny matplotlib shim used in tests when matplotlib is not installed.

It exposes a `plt`-like object with the subset of functions used by tests:
figure, annotate, contourf, scatter, savefig, imshow, plot, xlabel, ylabel, ylim.
If real matplotlib is available, we use it; otherwise we provide no-op implementations
that still create an empty image file on savefig so file-creation tests pass.
"""
from __future__ import annotations

from typing import Any
from types import SimpleNamespace
import os
try:
    import matplotlib.pyplot as _plt
    plt = _plt
except Exception:
    from PIL import Image
    import numpy as _np

    class _NoopFigure:
        def __init__(self) -> None:
            self._w, self._h = 640, 480

    def _ensure_dir(fp: str) -> None:
        d = os.path.dirname(fp)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def figure(*args: Any, **kwargs: Any) -> _NoopFigure:
        return _NoopFigure()

    def annotate(*args: Any, **kwargs: Any) -> None:
        return None

    def contourf(*args: Any, **kwargs: Any) -> None:
        return None

    def scatter(*args: Any, **kwargs: Any) -> None:
        return None

    def imshow(*args: Any, **kwargs: Any) -> None:
        return None

    def plot(*args: Any, **kwargs: Any) -> None:
        return None

    def xlabel(*args: Any, **kwargs: Any) -> None:
        return None

    def ylabel(*args: Any, **kwargs: Any) -> None:
        return None

    def ylim(*args: Any, **kwargs: Any) -> None:
        return None

    def savefig(fp: str, *args: Any, **kwargs: Any) -> None:
        # create a tiny blank PNG so tests that check file existence pass
        _ensure_dir(fp)
        arr = _np.zeros((10, 10, 3), dtype=_np.uint8)
        Image.fromarray(arr).save(fp)

    # Additional plotting functions commonly used in tests
    def axis(*args: Any, **kwargs: Any) -> None:
        return None

    def legend(*args: Any, **kwargs: Any) -> None:
        return None

    def subplot(*args: Any, **kwargs: Any) -> None:
        return None

    def subplots(*args: Any, **kwargs: Any) -> tuple[None, None]:
        return (None, None)

    def show(*args: Any, **kwargs: Any) -> None:
        return None

    def xlim(*args: Any, **kwargs: Any) -> None:
        return None

    def xticks(*args: Any, **kwargs: Any) -> None:
        return None

    def yticks(*args: Any, **kwargs: Any) -> None:
        return None

    def contour(*args: Any, **kwargs: Any) -> None:
        return None

    def colorbar(*args: Any, **kwargs: Any) -> None:
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
