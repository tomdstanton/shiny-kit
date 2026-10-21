"""Unit tests for shiny_kit core infrastructure: caching, warnings, progress, ui, widgets, and state."""

import enum
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from htmltools import Tag
from shiny import reactive

from shiny_kit.caching import BaseCache
from shiny_kit.progress import LazyShinyProgress, current_progress, progress_context
from shiny_kit.state import ArtifactRegistry, BaseSessionState, Requirement, artifact_registry
from shiny_kit.ui import EmptyState, Icon, KPICard, KPIRow, StagedList
from shiny_kit.warnings import WarningCatcher
from shiny_kit.widgets import DocstringParser, ParameterSpec, WidgetSelector

# ---------------------------------------------------------------------------
# 1. Caching (shiny_kit.caching)
# ---------------------------------------------------------------------------


@dataclass
class _SampleSpec:
    x: int
    y: str = "hello"
    hidden: int = field(default=99, repr=False)


class _CustomHashObj:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __cache_fingerprint__(self) -> str:
        return f"custom:{self.tag}"


class _UnserializableObj:
    pass


def test_base_cache_fingerprinting_and_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHINY_KIT_CACHE_DIR", str(tmp_path / "env_cache"))
    assert BaseCache.get_default_cache_dir() == (tmp_path / "env_cache").resolve()
    monkeypatch.delenv("SHINY_KIT_CACHE_DIR")
    monkeypatch.setenv("SHINY_KIT_HOME", str(tmp_path / "env_home"))
    assert BaseCache.get_default_cache_dir() == (tmp_path / "env_home" / "cache").resolve()

    fp1 = BaseCache.fingerprint(
        {
            "none": None,
            "bool": True,
            "int": 42,
            "float": 3.14,
            "bytes": b"raw",
            "path": Path("/tmp/a"),
            "dc": _SampleSpec(1, "a", hidden=10),
            "custom": _CustomHashObj("v1"),
            "set": {1, 2, 3},
            "seq": [1, (2, 3)],
            "obj": _UnserializableObj(),
        }
    )
    fp2 = BaseCache.fingerprint(
        {
            "none": None,
            "bool": True,
            "int": 42,
            "float": 3.14,
            "bytes": b"raw",
            "path": Path("/tmp/a"),
            "dc": _SampleSpec(1, "a", hidden=999),  # hidden has repr=False, so fingerprint must match
            "custom": _CustomHashObj("v1"),
            "set": {3, 2, 1},
            "seq": [1, (2, 3)],
            "obj": _UnserializableObj(),
        }
    )
    assert len(fp1) == 32
    assert isinstance(fp2, str)

    class SubCache(BaseCache):
        @classmethod
        def _update_custom_hasher(cls, hasher: Any, obj: Any) -> bool:
            if isinstance(obj, complex):
                hasher.update(f"complex:{obj.real},{obj.imag}".encode())
                return True
            return False

    assert SubCache.fingerprint(1 + 2j) != SubCache.fingerprint(1 + 3j)

    cache = BaseCache(cache_dir=tmp_path / "cache", max_memory_items=2, persist_to_disk=True)
    assert cache.get("missing") == (False, None)
    assert cache.misses == 1

    cache.put("k1", {"val": 1})
    cache.put("k2", {"val": 2})
    cache.put("k3", {"val": 3})  # Evicts k1 from in-memory LRU, but k1 remains on disk

    # Memory hit on k2
    hit, val = cache.get("k2")
    assert hit is True and val == {"val": 2}

    # Disk hit on k1 (re-promotes to LRU)
    hit, val = cache.get("k1")
    assert hit is True and val == {"val": 1}

    # Corrupt disk file recovery
    corrupt_path = cache._disk_path("corrupt_key")
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not-valid-pickle")
    assert cache.get("corrupt_key") == (False, None)
    assert not corrupt_path.exists()

    # call_or_load single-flight & caching
    calls = 0

    def compute(a: int, b: int = 1) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert cache.call_or_load(compute, 10, b=5) == 15
    assert cache.call_or_load(compute, 10, b=5) == 15
    assert calls == 1

    # Clear memory to force disk reload inside call_or_load lock
    cache._lru.clear()
    assert cache.call_or_load(compute, 10, b=5) == 15
    assert calls == 1

    # Failed disk dump cleans up tmp file
    class FailingCache(BaseCache):
        def _dump_to_disk(self, value: Any, path: Path) -> None:
            path.write_bytes(b"partial")
            raise OSError("disk full")

    fcache = FailingCache(cache_dir=tmp_path / "fcache")
    with pytest.raises(OSError, match="disk full"):
        fcache.put("bad", 123)

    cache.clear()
    assert cache.hits == 0 and cache.misses == 0
    assert not (cache.cache_dir / "artifacts").exists()


# ---------------------------------------------------------------------------
# 2. Warnings (shiny_kit.warnings)
# ---------------------------------------------------------------------------


def test_warning_catcher() -> None:
    with WarningCatcher(reemit=False) as catcher:
        warnings.warn("First user warning", UserWarning, stacklevel=1)
        warnings.warn("First user warning", UserWarning, stacklevel=1)
        warnings.warn("Ignored deprecation", DeprecationWarning, stacklevel=1)
        warnings.warn("Runtime alert", RuntimeWarning, stacklevel=1)

    assert catcher.messages == ("First user warning", "Runtime alert")
    assert WarningCatcher.merge([" a ", "b"], ("a", "", "c")) == ("a", "b", "c")

    # Test reemit=True
    with pytest.warns(UserWarning, match="Re-emitted warning"), WarningCatcher(reemit=True) as catcher2:
        warnings.warn("Re-emitted warning", UserWarning, stacklevel=1)
    assert catcher2.messages == ("Re-emitted warning",)

    # Site-packages non-UserWarning is filtered out
    fake_msg = warnings.WarningMessage(
        message=RuntimeWarning("internal"),
        category=RuntimeWarning,
        filename="/env/lib/python3.14/site-packages/pkg/mod.py",
        lineno=10,
    )
    assert WarningCatcher.is_relevant(fake_msg) is False


# ---------------------------------------------------------------------------
# 3. Progress (shiny_kit.progress)
# ---------------------------------------------------------------------------


def test_progress_reporter_and_lazy_shiny_progress() -> None:
    # Default no-op reporter outside context
    noop = current_progress()
    assert noop.is_active is False
    assert noop.active is False
    noop.report(0.5, "ignored")
    with noop.subrange(0.0, 1.0) as sub_noop:
        assert sub_noop is noop

    events: list[tuple[float, str | None, str | None]] = []
    with progress_context(
        lambda f, m, d: events.append((round(f, 3), m, d)),
        message="Starting",
        min_interval_sec=10.0,
    ) as root:
        assert root.is_active is True
        root.report(0.2, message="Step 1", detail="init", force=True)
        # Throttled because min_interval_sec=10.0 and force=False and fraction < 1.0
        root.report(0.4, detail="throttled")
        with root.subrange(0.5, 1.0, detail_prefix="Fold 1") as sub1:
            with sub1.subrange(0.0, 0.5, message="Nested", detail_prefix="Inner") as sub2:
                sub2.report(0.5, detail="half", force=True)
            sub1.report(1.0, force=True)

    assert events == [
        (0.2, "Step 1", "init"),
        (0.625, "Nested", "Fold 1 · Inner · half"),
        (1.0, "Step 1", "Fold 1"),
    ]

    # LazyShinyProgress with immediate flush (grace_sec=0) and custom factory
    mock_bar = MagicMock()
    loop = MagicMock()
    loop.call_soon_threadsafe.side_effect = lambda fn: fn()

    lazy = LazyShinyProgress(
        None,
        default_message="Init",
        grace_sec=0.0,
        loop=loop,
        progress_factory=lambda: mock_bar,
    )
    assert mock_bar.set.called
    lazy.update(0.6, "Running", "60%")
    mock_bar.set.assert_called_with(value=0.6, message="Running", detail="60%")
    lazy.close()
    assert mock_bar.close.called
    # Subsequent update/close are no-ops
    lazy.update(0.9, "After close", None)
    lazy.close()

    # LazyShinyProgress with grace_sec > 0 and timer cancellation before grace expires
    timer_handle = MagicMock()
    loop2 = MagicMock()
    loop2.call_later.return_value = timer_handle
    mock_sess = MagicMock()
    with patch("shiny.ui.Progress") as mock_prog_cls:
        lazy2 = LazyShinyProgress(mock_sess, grace_sec=5.0, loop=loop2)
        lazy2.update(0.1, None, "early")
        # Trigger flush manually as if timer fired
        lazy2._flush_on_loop()
        assert mock_prog_cls.called
        lazy2.close()


# ---------------------------------------------------------------------------
# 4. UI Primitives (shiny_kit.ui)
# ---------------------------------------------------------------------------


def test_ui_primitives() -> None:
    with reactive.isolate():
        assert Icon.lucide(None) is None
        svg = Icon.lucide("arrow_right", size=20, class_="custom-cls")
        assert svg is not None
        assert 'xmlns="http://www.w3.org/2000/svg"' in str(svg)
        assert "custom-cls" in str(svg)

        with pytest.raises(ValueError, match="does not exist in Lucide"):
            Icon.lucide("definitely-not-a-valid-lucide-icon-xyz")

        empty = EmptyState(icon="database", title="Empty", message="Nothing here", dashed=True, compact=True)
        html_empty = str(empty())
        assert "border-dashed" in html_empty and "py-3" in html_empty

        row = KPIRow(
            KPICard(label="N", value="100", subtitle="samples", truncate_subtitle=True),
            KPICard(label="Rate", value="0.5", subtitle="ci", tooltip="95% CI"),
        )
        html_row = str(row.ui())
        assert "100" in html_row and "text-truncate" in html_row and "95% CI" in html_row

        staged = StagedList[str](["a"])
        assert not staged.is_empty
        assert len(staged) == 1
        staged.add("b", deduplicate=True)
        staged.add("a", deduplicate=True)
        assert staged.items == ["b", "a"]
        staged.remove_at(0)
        assert staged.items == ["a"]
        assert staged.pop() == "a"
        assert staged.pop() is None
        assert staged.is_empty

        empty_badges = str(staged.render_badges(lambda i, x: f"{i}:{x}"))
        assert "No items staged yet." in empty_badges

        staged.set(["x", "y"])
        populated_badges = str(staged.render_badges(lambda i, x: f"{i}:{x}", wrapper_class="d-flex"))
        assert "1:x" in populated_badges and "2:y" in populated_badges

        btns = str(StagedList.action_buttons("add_btn", pop_id="pop_btn", clear_id="clr_btn"))
        assert "add_btn" in btns and "pop_btn" in btns and "clr_btn" in btns
        staged.clear()
        assert staged.is_empty


# ---------------------------------------------------------------------------
# 5. Widgets & DocstringParser (shiny_kit.widgets)
# ---------------------------------------------------------------------------


class _ColorMode(enum.Enum):
    FAST = "fast"
    EXACT = "exact"


class _BaseEstimator:  # noqa: B903
    """Base estimator class.

    Parameters
    ----------
    seed : int
        Random number seed for reproducibility.
    """

    def __init__(self, seed: int = 42, rng: complex | None = None) -> None:
        self.seed = seed
        self.rng = rng


class _ChildEstimator(_BaseEstimator):
    """Child estimator with Google docstring.

    Args:
        alpha (float): Significance level
            spanning multiple lines.
        mode: Execution mode.
        flag: Boolean toggle.
        label: Optional label text.
        count: Optional integer count.
        ratio: Optional float ratio.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        mode: Literal["a", "b"] | None = "a",
        kind: _ColorMode = _ColorMode.FAST,
        flag: bool = True,
        label: str | None = None,
        count: int | None = None,
        ratio: float | None = None,
        callback: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)


def test_docstring_parser_and_widget_selector() -> None:
    docs = DocstringParser.parse_parameters(_ChildEstimator)
    assert "Significance level spanning multiple lines." in docs["alpha"]
    assert "Random number seed" in docs["seed"]
    # Cached hit
    assert DocstringParser.parse_parameters(_ChildEstimator) == docs

    class CustomSelector(WidgetSelector):
        default_exclude = {"callback"}
        non_ui_types = (complex,)

    sel = CustomSelector.default()
    assert CustomSelector.default() is sel

    specs = sel.introspect(_ChildEstimator)
    names = [s.name for s in specs]
    assert "seed" in names
    assert "callback" not in names

    # Register custom rule for parameter named "seed"
    @sel.register(lambda s: s.name == "seed")
    def _build_seed(input_id: str, spec: ParameterSpec) -> Tag:
        return Tag("div", f"custom-{input_id}")

    widgets = sel.build_form(
        _ChildEstimator,
        choices={"label": {"opt1": "Option 1"}, "mode": ["a", "b"]},
        prefix="p_",
    )
    assert len(widgets) == len(specs)
    assert any("custom-p_seed" in str(w) for w in widgets)

    # Extract kwargs with various input representations
    extracted = sel.extract_kwargs(
        _ChildEstimator,
        {
            "p_alpha": lambda: "0.01",
            "p_mode": "",
            "p_kind": "EXACT",
            "p_flag": 0,
            "p_label": "custom",
            "p_count": "",
            "p_ratio": "1.5",
            "p_seed": "7",
        },
        prefix="p_",
    )
    assert extracted["alpha"] == 0.01
    assert extracted["mode"] is None
    assert extracted["kind"] == _ColorMode.EXACT
    assert extracted["flag"] is False
    assert extracted["label"] == "custom"
    assert extracted["count"] is None
    assert extracted["ratio"] == 1.5
    assert extracted["seed"] == 7

    # Also test Enum extraction by value and direct Enum instance
    ext2 = sel.extract_kwargs(_ChildEstimator, {"kind": "fast"})
    assert ext2["kind"] == _ColorMode.FAST
    ext3 = sel.extract_kwargs(_ChildEstimator, {"kind": _ColorMode.EXACT})
    assert ext3["kind"] == _ColorMode.EXACT


# ---------------------------------------------------------------------------
# 6. State & ArtifactRegistry (shiny_kit.state)
# ---------------------------------------------------------------------------


@dataclass
class _DummyItem:
    name: str
    size: int = 3

    def __len__(self) -> int:
        return self.size


class _DemoState(BaseSessionState):
    datasets = artifact_registry[_DummyItem](
        "Datasets",
        icon="database",
        toast="Saved {name} ({rows} rows)",
        on_register="on_ds_added",
        on_select=lambda state, item, name: state.preferences.__setitem__("selected", str(name)),
        on_remove=lambda state, item, **kw: state.preferences.__setitem__("removed", item.name),
    )
    models = artifact_registry[dict[str, int]](
        icon=None,
        show_in_navbar=False,
        namer=lambda d: f"model_{d.get('id', 0)}",
        toast=lambda d: f"Model #{d.get('id', 0)} ready",
    )

    def __init__(self) -> None:
        super().__init__()
        self.added_log: list[str] = []
        self.reset_called: bool = False

    def on_ds_added(self, item: _DummyItem, name: str) -> None:
        self.added_log.append(name)

    def on_reset(self) -> None:
        self.reset_called = True


def test_state_and_artifact_registry() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseSessionState"):
        BaseSessionState()

    assert isinstance(_DemoState.datasets, artifact_registry)

    with reactive.isolate():
        state = _DemoState()
        assert state.color_mode == "light"
        state.color_mode = "dark"
        assert state.color_mode == "dark"

        req_rule = Requirement("datasets", min_count=1, target_tab="tab_ds")
        assert req_rule.display_label == "Datasets"
        assert not req_rule.is_met(state)
        assert req_rule.status_text(state) == "Datasets: 0 / 1 available"

        # Register via 1-arg and 2-arg overloads
        item1 = _DummyItem("cohort_a", size=5)
        state.register_artifact("datasets", item1, name="  cohort_1  ")
        assert item1.name == "cohort_1"
        assert state.added_log == ["cohort_1"]
        assert state.datasets.count == 1
        assert state.datasets.active == item1
        assert state.datasets.active_key == "cohort_1"
        assert state.datasets.version == 1
        assert req_rule.is_met(state)

        item2 = _DummyItem("cohort_2", size=8)
        state.datasets.register("explicit_2", item2, set_active=False)
        assert state.datasets.active_key == "cohort_1"
        assert state.datasets.names == ["cohort_1", "explicit_2"]

        state.datasets.active_key = "explicit_2"
        assert state.preferences["selected"] == "explicit_2"

        with pytest.raises(KeyError):
            state.datasets.select("nonexistent")

        with pytest.raises(TypeError):
            state.datasets.register()  # ty: ignore[no-matching-overload]

        empty_reg = ArtifactRegistry[_DummyItem]("EmptyReg", namer=lambda _: "   ")
        with pytest.raises(ValueError, match="cannot be empty"):
            empty_reg.register(item1)

        # Toast formatting
        assert state.format_task_success(item1, registry_name="datasets") == "Saved cohort_1 (5 rows)"
        m1 = {"id": 7}
        state.models.register(m1)
        assert state.format_task_success(m1) == "Model #7 ready"

        # Navbar badges & summary counts
        assert state.summary_counts() == {"datasets": 2, "models": 1}
        badges_html = str(state.render_navbar_badges())
        assert "Datasets: 2" in badges_html

        # Remove active item updates active_key to remaining item
        state.datasets.remove("explicit_2")
        assert state.preferences["removed"] == "cohort_2"
        assert state.datasets.active_key == "cohort_1"
        state.datasets.remove("missing_key")  # no-op

        with pytest.raises(AttributeError, match="no ArtifactRegistry"):
            state.get_registry("unknown_reg")

        # Session binding and navigation
        mock_sess = MagicMock()
        mock_sess.id = "sess-123"
        mock_sess._root_session = mock_sess
        s1 = _DemoState.from_session(mock_sess)
        s2 = _DemoState.from_session(mock_sess)
        assert s1 is s2
        assert mock_sess.on_ended.called
        cleanup_cb = mock_sess.on_ended.call_args[0][0]

        with patch("shiny.ui.update_navset") as mock_nav:
            s1.navigate_to("target_tab")
            mock_nav.assert_called_once_with("main_nav", selected="target_tab", session=mock_sess)

        cleanup_cb()
        assert "sess-123" not in _DemoState._registry

        state.reset_all()
        assert state.reset_called is True
        assert state.datasets.count == 0
