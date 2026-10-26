"""Pure-stdlib two-tier content-addressable cache with LRU memory and pluggable disk persistence."""

import asyncio
import copy
import hashlib
import json
import os
import pickle
import shutil
import threading
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from shiny_kit.warnings import WarningCatcher


class _SyncKeyLock:
    """Weakref-capable wrapper around `threading.Lock` for automatic single-flight lock GC."""

    __slots__ = ("__weakref__", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> bool:
        return self._lock.acquire()

    def __exit__(self, *args: object) -> None:
        self._lock.release()


class BaseCache:
    """Thread-safe two-tier content-addressable cache (in-memory LRU + pluggable disk tier).

    Provides deterministic SHA-256 fingerprinting across primitives, paths, dataclasses,
    collections, and objects implementing `__cache_fingerprint__()`, alongside single-flight
    per-key locking so concurrent callers computing the same key wait on the first computation.

    Subclasses can override `_update_custom_hasher()` to fingerprint domain-specific types
    (e.g. DataFrames or ML estimators) and `_dump_to_disk()` / `_load_from_disk()` for custom
    serialization formats (e.g. `joblib`).

    Attributes:
        disk_extension (ClassVar[str]): File extension used for disk cache entries. Defaults to `".pkl"`.
        env_prefix (ClassVar[str]): Environment variable prefix (`{env_prefix}_CACHE_DIR`, `{env_prefix}_HOME`).
        default_home_dirname (ClassVar[str]): Default home directory folder name. Defaults to `".shiny-kit"`.
    """

    disk_extension: ClassVar[str] = ".pkl"
    env_prefix: ClassVar[str] = "SHINY_KIT"
    default_home_dirname: ClassVar[str] = ".shiny-kit"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        max_memory_items: int = 64,
        persist_to_disk: bool = True,
    ) -> None:
        """Initializes `BaseCache` with an in-memory LRU and optional disk directory.

        Args:
            cache_dir: Optional directory path for persistent disk storage.
            max_memory_items: Maximum number of items retained in the in-memory LRU tier.
            persist_to_disk: Whether to persist cached values to disk.
        """
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.get_default_cache_dir()
        self.max_memory_items = max_memory_items
        self.persist_to_disk = persist_to_disk
        self._lru: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._inflight_locks: weakref.WeakValueDictionary[str, _SyncKeyLock] = weakref.WeakValueDictionary()
        self._async_inflight_locks: weakref.WeakValueDictionary[tuple[int, str], asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self.hits: int = 0
        self.misses: int = 0

    @classmethod
    def get_default_cache_dir(cls) -> Path:
        """Resolves the persistent cache directory from environment variables or `~/.shiny-kit/cache`."""
        if custom := os.environ.get(f"{cls.env_prefix}_CACHE_DIR"):
            return Path(custom).expanduser().resolve()
        base = Path(os.environ.get(f"{cls.env_prefix}_HOME", f"~/{cls.default_home_dirname}")).expanduser().resolve()
        return base / "cache"

    @classmethod
    def fingerprint(cls, obj: Any) -> str:
        """Computes a deterministic 32-character SHA-256 hex digest for `obj`.

        Args:
            obj: Arbitrary Python object, dataclass, mapping, sequence, or domain object.

        Returns:
            32-character hexadecimal SHA-256 fingerprint string.
        """
        hasher = hashlib.sha256()
        cls._update_hasher(hasher, obj)
        return hasher.hexdigest()[:32]

    @staticmethod
    def _is_dataframe(obj: Any) -> bool:
        """Returns True if `obj` duck-types as a Polars or Pandas DataFrame."""
        if not (hasattr(obj, "shape") and hasattr(obj, "columns")):
            return False
        return isinstance(getattr(obj, "schema", None), Mapping) or hasattr(obj, "dtypes")

    @staticmethod
    def _update_dataframe_hasher(hasher: "hashlib._Hash", df: Any) -> None:
        """Hashes a DataFrame using native vectorized row hashing when available, with structural fallback."""
        schema_map = getattr(df, "schema", None)
        if isinstance(schema_map, Mapping):
            schema_repr = [(col, str(dtype)) for col, dtype in schema_map.items()]
        else:
            dtypes = getattr(df, "dtypes", None)
            schema_repr = (
                [(col, str(dtype)) for col, dtype in dtypes.items()]
                if hasattr(dtypes, "items")
                else list(getattr(df, "columns", ()))
            )
        hasher.update(f"DF:{df.shape}:{schema_repr}:".encode())
        if len(df) == 0:
            return
        if hasattr(df, "hash_rows") and callable(df.hash_rows):
            row_hashes = df.hash_rows(seed=0)
            if hasattr(row_hashes, "to_numpy"):
                hasher.update(memoryview(row_hashes.to_numpy()))
                return
            hasher.update(f"hash_sum:{row_hashes.sum()}:{row_hashes[0]}:{row_hashes[-1]};".encode())
            return
        sample_col = df.columns[0]
        col_s = df[sample_col]
        head_vals = col_s.head(5).to_list() if hasattr(col_s, "head") else list(col_s[:5])
        tail_vals = col_s.tail(5).to_list() if hasattr(col_s, "tail") else list(col_s[-5:])
        hasher.update(f"head:{head_vals}:tail:{tail_vals}:".encode())

    @classmethod
    def _update_custom_hasher(cls, hasher: "hashlib._Hash", obj: Any) -> bool:
        """Hook for subclasses to hash custom domain types before fallback structural hashing.

        Args:
            hasher: Active `hashlib` SHA-256 hasher.
            obj: Object being fingerprinted.

        Returns:
            `True` if `obj` was handled by this hook, `False` to proceed with default rules.
        """
        return False

    @classmethod
    def _update_hasher(cls, hasher: "hashlib._Hash", obj: Any) -> None:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            hasher.update(f"{type(obj).__qualname__}:{obj!r};".encode())
            return

        if isinstance(obj, bytes):
            hasher.update(b"bytes:")
            hasher.update(hashlib.sha256(obj).digest())
            return

        if isinstance(obj, Path):
            hasher.update(f"Path:{obj};".encode())
            return

        if hasattr(obj, "__cache_fingerprint__") and callable(obj.__cache_fingerprint__):
            cls._update_hasher(hasher, obj.__cache_fingerprint__())
            return

        if cls._update_custom_hasher(hasher, obj):
            return

        if cls._is_dataframe(obj):
            cls._update_dataframe_hasher(hasher, obj)
            return

        if hasattr(obj, "data") and hasattr(obj, "indices") and hasattr(obj, "indptr") and hasattr(obj, "shape"):
            hasher.update(f"Sparse:{type(obj).__qualname__}:{obj.shape}:".encode())
            cls._update_hasher(hasher, obj.data)
            cls._update_hasher(hasher, obj.indices)
            cls._update_hasher(hasher, obj.indptr)
            return

        if hasattr(obj, "tobytes") and hasattr(obj, "shape") and hasattr(obj, "dtype") and callable(obj.tobytes):
            hasher.update(f"Array:{obj.shape}:{obj.dtype}:".encode())
            hasher.update(hashlib.sha256(obj.tobytes()).digest())
            return

        if hasattr(obj, "get_params") and callable(obj.get_params) and not isinstance(obj, type):
            hasher.update(f"Estimator:{type(obj).__module__}.{type(obj).__qualname__}:".encode())
            try:
                params = obj.get_params(deep=False)
            except TypeError:
                params = obj.get_params()
            if isinstance(params, Mapping):
                for k in sorted(params.keys(), key=str):
                    hasher.update(f"param:{k}=".encode())
                    cls._update_hasher(hasher, params[k])
            obj_vars = getattr(obj, "__dict__", None)
            if isinstance(obj_vars, Mapping):
                for fitted_attr in sorted(
                    k for k in obj_vars if isinstance(k, str) and k.endswith("_") and not k.startswith("_")
                ):
                    val = obj_vars[fitted_attr]
                    if val is not None and cls._is_dataframe(val):
                        hasher.update(f"fitted:{fitted_attr}:".encode())
                        cls._update_dataframe_hasher(hasher, val)
            return

        if is_dataclass(obj) and not isinstance(obj, type):
            hasher.update(f"Dataclass:{type(obj).__qualname__}:".encode())
            for f in fields(obj):
                if not f.repr:
                    continue
                hasher.update(f"{f.name}=".encode())
                cls._update_hasher(hasher, getattr(obj, f.name))
            return

        if isinstance(obj, Mapping):
            hasher.update(b"Map:{")
            for k in sorted(obj.keys(), key=str):
                cls._update_hasher(hasher, k)
                hasher.update(b"->")
                cls._update_hasher(hasher, obj[k])
            hasher.update(b"}")
            return

        if isinstance(obj, (Sequence, set, frozenset)) and not isinstance(obj, (str, bytes)):
            seq = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
            hasher.update(b"Seq:[")
            for item in seq:
                cls._update_hasher(hasher, item)
            hasher.update(b"]")
            return

        try:
            hasher.update(json.dumps(obj, sort_keys=True, default=repr).encode())
        except TypeError:
            hasher.update(repr(obj).encode())

    def _get_key_lock(self, key: str) -> _SyncKeyLock:
        with self._lock:
            lock = self._inflight_locks.get(key)
            if lock is None:
                lock = _SyncKeyLock()
                self._inflight_locks[key] = lock
            return lock

    def _get_async_key_lock(self, key: str) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        compound = (loop_id, key)
        with self._lock:
            lock = self._async_inflight_locks.get(compound)
            if lock is None:
                lock = asyncio.Lock()
                self._async_inflight_locks[compound] = lock
            return lock

    def _call_key(self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        fn_id = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}"
        fn_self = getattr(fn, "__self__", None)
        if fn_self is not None and not isinstance(fn_self, type):
            return self.fingerprint(("call", fn_id, fn_self, args, kwargs))
        return self.fingerprint(("call", fn_id, args, kwargs))

    def _lru_get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return True, self._lru[key]
            return False, None

    def _lru_put(self, key: str, value: Any) -> None:
        with self._lock:
            self._lru[key] = value
            self._lru.move_to_end(key)
            while len(self._lru) > self.max_memory_items:
                self._lru.popitem(last=False)

    def _disk_path(self, key: str) -> Path:
        return self.cache_dir / "artifacts" / f"{key}{self.disk_extension}"

    def _dump_to_disk(self, value: Any, path: Path) -> None:
        """Serializes `value` to `path`. Override in subclasses for custom formats like `joblib`."""
        with path.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_disk(self, path: Path) -> Any:
        """Deserializes a cached artifact from `path`. Override in subclasses for custom formats."""
        with path.open("rb") as fh:
            return pickle.load(fh)

    def get(self, key: str) -> tuple[bool, Any]:
        """Looks up a cached artifact by fingerprint key across memory and disk tiers.

        Args:
            key: 32-character hexadecimal fingerprint key.

        Returns:
            Tuple of `(hit, value)` where `hit` is `True` if found in memory or disk cache.
        """
        with self._lock:
            hit, val = self._lru_get(key)
            if hit:
                self.hits += 1
                return True, val

        if self.persist_to_disk:
            path = self._disk_path(key)
            if path.exists():
                try:
                    val = self._load_from_disk(path)
                    with self._lock:
                        self._lru_put(key, val)
                        self.hits += 1
                    return True, val
                except Exception:
                    path.unlink(missing_ok=True)

        with self._lock:
            self.misses += 1
        return False, None

    def put(self, key: str, value: Any) -> Any:
        """Stores an artifact in the in-memory LRU and optional disk tier atomically.

        Args:
            key: 32-character hexadecimal fingerprint key.
            value: Artifact to cache.

        Returns:
            The stored `value`.
        """
        self._lru_put(key, value)
        if self.persist_to_disk:
            path = self._disk_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{uuid4().hex[:8]}")
            try:
                self._dump_to_disk(value, tmp_path)
                tmp_path.replace(path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        return value

    def call_or_load[R](
        self,
        fn: Callable[..., R],
        *args: Any,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> R:
        """Executes `fn(*args, **kwargs)` with thread-safe two-tier fingerprint caching.

        Args:
            fn: Callable to execute on cache miss.
            *args: Positional arguments forwarded to `fn`.
            force_refresh: If `True`, bypasses cached values and recomputes `fn`.
            **kwargs: Keyword arguments forwarded to `fn`.

        Returns:
            Cached or freshly computed return value `R`.
        """
        key = self._call_key(fn, args, kwargs)
        if not force_refresh:
            with self._lock:
                hit, cached = self._lru_get(key)
                if hit:
                    self.hits += 1
                    return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            if not force_refresh:
                hit, cached = self.get(key)
                if hit:
                    return cached
            res = fn(*args, **kwargs)
            return self.put(key, res)

    async def acall_or_load[R](
        self,
        fn: Callable[..., Awaitable[R]],
        *args: Any,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> R:
        """Asynchronously executes `await fn(*args, **kwargs)` with single-flight fingerprint caching.

        Args:
            fn: Async callable to await on cache miss.
            *args: Positional arguments forwarded to `fn`.
            force_refresh: If `True`, bypasses cached values and recomputes `fn`.
            **kwargs: Keyword arguments forwarded to `fn`.

        Returns:
            Cached or freshly computed return value `R`.
        """
        key = self._call_key(fn, args, kwargs)
        if not force_refresh:
            with self._lock:
                hit, cached = self._lru_get(key)
                if hit:
                    self.hits += 1
                    return cached

        async_lock = self._get_async_key_lock(key)
        async with async_lock:
            if not force_refresh:
                hit, cached = self.get(key)
                if hit:
                    return cached
            res = await fn(*args, **kwargs)
            return self.put(key, res)

    def compute_fit_key(self, estimator: Any, X: Any, y: Any = None) -> str:
        """Computes a composite cache key for fitting `estimator` on `(X, y)`.

        Args:
            estimator: Unfitted or configured estimator instance.
            X: Input data object.
            y: Optional target argument.

        Returns:
            32-character SHA-256 fingerprint string.
        """
        return self.fingerprint(("fit", estimator, X, y))

    @staticmethod
    def _clone_estimator[E](estimator: E) -> E:
        """Clones an estimator using `__sklearn_clone__()`, `get_params()`, or shallow copy without importing `sklearn`."""
        if hasattr(estimator, "__sklearn_clone__") and callable(estimator.__sklearn_clone__):
            return estimator.__sklearn_clone__()  # ty: ignore[invalid-return-type]
        if hasattr(estimator, "get_params") and callable(estimator.get_params):
            try:
                params = estimator.get_params(deep=False)
                if isinstance(params, dict):
                    return type(estimator)(**{k: copy.deepcopy(v) for k, v in params.items()})
            except Exception:
                pass
        return copy.copy(estimator)

    def fit_or_load[E](self, estimator: E, X: Any = None, y: Any = None) -> E:
        """Returns a cached fitted clone of `estimator` on `(X, y)`, or fits and caches it on miss.

        Uses per-key single-flight locking so concurrent sessions fitting the same model and
        dataset wait on the first fit and share its cached output without redundant execution.
        Captures and re-emits user-facing warnings via `WarningCatcher`.

        Args:
            estimator: Unfitted or configured estimator implementing `.fit(X)` / `.fit(X, y)`.
            X: Input data (or `None` when wrapping pre-fitted models).
            y: Optional target argument.

        Returns:
            The fitted estimator instance.
        """
        key = self.compute_fit_key(estimator, X, y)
        with self._lock:
            hit, cached = self._lru_get(key)
            if hit:
                self.hits += 1
                WarningCatcher.emit_messages(getattr(cached, "warnings_", ()), stacklevel=2)
                return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            hit, cached = self.get(key)
            if hit:
                WarningCatcher.emit_messages(getattr(cached, "warnings_", ()), stacklevel=2)
                return cached

            fitted: E = self._clone_estimator(estimator)
            with WarningCatcher() as catcher:
                if y is not None:
                    fitted.fit(X, y)  # ty: ignore[unresolved-attribute]
                else:
                    fitted.fit(X)  # ty: ignore[unresolved-attribute]
            inner_warnings = getattr(getattr(fitted, "estimator_", None), "warnings_", ())
            own_warnings = getattr(fitted, "warnings_", ())
            merged_warnings = WarningCatcher.merge(inner_warnings, own_warnings, catcher.messages)
            try:
                fitted.warnings_ = merged_warnings  # ty: ignore[unresolved-attribute]
            except Exception:
                pass
            stored = self.put(key, fitted)
            WarningCatcher.emit_messages(merged_warnings, stacklevel=2)
            return stored

    def clear(self) -> None:
        """Clears both the in-memory LRU cache and persistent disk artifacts."""
        with self._lock:
            self._lru.clear()
            self._inflight_locks.clear()
            self._async_inflight_locks.clear()
            self.hits = 0
            self.misses = 0
            artifacts_dir = self.cache_dir / "artifacts"
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir, ignore_errors=True)


default_cache = BaseCache(persist_to_disk=False)
