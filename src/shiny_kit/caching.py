"""Pure-stdlib two-tier content-addressable cache with LRU memory and pluggable disk persistence."""

import hashlib
import json
import os
import pickle
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4


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
        self._inflight_locks: dict[str, threading.Lock] = {}
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

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._inflight_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._inflight_locks[key] = lock
            return lock

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

    def call_or_load[R](self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Executes `fn(*args, **kwargs)` with thread-safe two-tier fingerprint caching.

        Args:
            fn: Callable to execute on cache miss.
            *args: Positional arguments forwarded to `fn`.
            **kwargs: Keyword arguments forwarded to `fn`.

        Returns:
            Cached or freshly computed return value `R`.
        """
        fn_id = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}"
        key = self.fingerprint(("call", fn_id, args, kwargs))
        with self._lock:
            hit, cached = self._lru_get(key)
            if hit:
                self.hits += 1
                return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            hit, cached = self.get(key)
            if hit:
                return cached
            res = fn(*args, **kwargs)
            return self.put(key, res)

    def clear(self) -> None:
        """Clears both the in-memory LRU cache and persistent disk artifacts."""
        with self._lock:
            self._lru.clear()
            self._inflight_locks.clear()
            self.hits = 0
            self.misses = 0
            artifacts_dir = self.cache_dir / "artifacts"
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir, ignore_errors=True)


default_cache = BaseCache()
