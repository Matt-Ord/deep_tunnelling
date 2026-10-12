"""Caching utilities (from classical_diffusion)."""

from __future__ import annotations

import pickle  # ruff:ignore[suspicious-pickle-import]
import types
from contextlib import contextmanager
from contextvars import ContextVar
from functools import update_wrapper
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

CallType = Literal[
    "load_or_call_cached",
    "load_or_call_uncached",
    "call_uncached",
    "call_cached",
]


def _reduce_typevars(obj: object) -> tuple[type, tuple[str]]:
    # This tells pickle: "If you see a TypeVar, just treat it as its name string"
    # This prevents the 'typing.M' lookup entirely.
    return str, (str(obj),)


_cache_base_path = ContextVar[Path | None]("cache_base_path", default=Path("data"))


@contextmanager
def cache_base_path(path: Path) -> Generator[None]:
    """Context manager to temporarily set the cache base path."""
    token = _cache_base_path.set(path)
    try:
        yield
    finally:
        _cache_base_path.reset(token)


class CachedFunction[**P, R]:
    """A function wrapper which is used to cache the output."""

    def __init__(
        self,
        function: Callable[P, R],
        path: Path | Callable[P, Path | None] | None,
        *,
        default_call: CallType = "load_or_call_cached",
    ) -> None:
        self._inner = function
        self.Path = path

        self.default_call: CallType = default_call

    def _get_cache_path(self, *args: P.args, **kw: P.kwargs) -> Path | None:
        cache_path = self.Path(*args, **kw) if callable(self.Path) else self.Path  # ty:ignore[call-top-callable]
        base_path = _cache_base_path.get()
        if cache_path is None or base_path is None:
            return None
        return base_path / cache_path  # ty: ignore[unsupported-operator]

    def call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, without using the cache."""
        return self._inner(*args, **kw)

    def call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function, and save the result to the cache."""
        obj = self.call_uncached(*args, **kw)
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is not None:
            buffers = list[pickle.PickleBuffer]()
            with cache_path.open("wb") as f:
                pickler = pickle.Pickler(
                    f,
                    pickle.HIGHEST_PROTOCOL,
                    buffer_callback=buffers.append,
                )
                pickler.dispatch_table = pickle.dispatch_table.copy()  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                pickler.dispatch_table[TypeVar] = _reduce_typevars  # type: ignore[attr-defined]
                pickler.dispatch_table[types.GenericAlias] = _reduce_typevars  # type: ignore[attr-defined]
                pickler.dump(obj)

            np.savez(cache_path.with_suffix(".buffer.npz"), *buffers)
        return obj

    def _load_cache(self, *args: P.args, **kw: P.kwargs) -> R | None:
        """Load data from cache."""
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is None:
            return None

        try:  # ruff:ignore[too-many-statements-in-try-clause]
            buffer_data = np.load(cache_path.with_suffix(".buffer.npz"))

            def _get_buffer() -> Generator[memoryview, memoryview]:
                i = 0
                while True:
                    try:
                        yield buffer_data[f"arr_{i}"]
                    except KeyError:
                        break
                    i += 1

            buffers = _get_buffer()
        except FileNotFoundError:
            buffers = list[pickle.PickleBuffer]()

        try:
            with cache_path.open("rb") as f:
                unpickler = pickle.Unpickler(f, buffers=buffers)  # ruff:ignore[suspicious-pickle-usage]
                return unpickler.load()
        except FileNotFoundError:
            return None

    def load_or_call_uncached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function uncached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_uncached(*args, **kw)
        return obj

    def load_or_call_cached(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function cached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_cached(*args, **kw)
        return obj

    def __call__(self, *args: P.args, **kw: P.kwargs) -> R:
        """Call the function using the cache."""
        match self.default_call:
            case "call_cached":
                return self.call_cached(*args, **kw)
            case "call_uncached":
                return self.call_uncached(*args, **kw)
            case "load_or_call_cached":
                return self.load_or_call_cached(*args, **kw)
            case "load_or_call_uncached":
                return self.load_or_call_uncached(*args, **kw)


@overload
def cached[**P, R](
    path: Path | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


@overload
def cached[**P, R](
    path: Callable[P, Path | None],
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


def cached[**P, R](
    path: Path | Callable[P, Path | None] | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]:
    """
    Cache the response of the function at the given path using pickle.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _cached(f: Callable[P, R]) -> CachedFunction[P, R]:
        return update_wrapper(
            CachedFunction(f, path, default_call=default_call),
            f,
        )  # ty: ignore[invalid-return-type]

    return _cached
