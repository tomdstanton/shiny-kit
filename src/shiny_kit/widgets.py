"""Introspective widget builder (`WidgetSelector`) and docstring parser (`DocstringParser`)."""

import inspect
import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, Self, get_args, get_origin

from htmltools import TagChild
from shiny import Inputs
from shiny import ui as shiny_ui


class DocstringParser:
    """Extracts parameter documentation from Google-style (`Args:`) and NumPy-style (`Parameters`) docstrings."""

    _CACHE: ClassVar[dict[type[Any], dict[str, str]]] = {}

    _NUMPY_SECTION_RE = re.compile(
        r"Parameters\s*\n\s*-+\s*\n(?P<content>.*?)(?=\n\s*(?:Returns|Attributes|Raises|See Also|Notes|References|Examples)\s*\n\s*-+|$)",
        re.DOTALL,
    )
    _NUMPY_ITEM_RE = re.compile(
        r"^(?P<name>\w+)\s*(?::\s*(?P<type>[^\n]+))?\n(?P<desc>(?:[ \t]+[^\n]*\n?)+)",
        re.MULTILINE,
    )
    _GOOGLE_SECTION_RE = re.compile(
        r"(?:Args|Arguments|Parameters):\s*\n(?P<content>.*?)(?=\n\s*(?:Returns|Yields|Raises|Attributes|Note|Notes|Example|Examples):\s*\n|$)",
        re.DOTALL,
    )
    _GOOGLE_ITEM_RE = re.compile(
        r"^[ \t]+(?P<name>\w+)(?:\s*\([^)]*\))?:\s*(?P<first_line>[^\n]*)(?P<cont>(?:\n[ \t]{6,}[^\n]*)*)",
        re.MULTILINE,
    )

    @classmethod
    def parse_parameters(cls, target_cls: type[Any]) -> dict[str, str]:
        """Parses and caches parameter descriptions from a class and its MRO hierarchy docstrings.

        Args:
            target_cls: Target class whose MRO docstrings should be inspected.

        Returns:
            Dictionary mapping parameter names to extracted one-line or multi-line descriptions.
        """
        cached = cls._CACHE.get(target_cls)
        if cached is not None:
            return dict(cached)

        docs_to_search: list[str | None] = []
        for base in target_cls.__mro__:
            if base is object:
                continue
            docs_to_search.append(inspect.getdoc(base))
            if "__init__" in base.__dict__:
                docs_to_search.append(inspect.getdoc(base.__init__))

        params: dict[str, str] = {}
        for doc in docs_to_search:
            if not doc:
                continue

            if g_match := cls._GOOGLE_SECTION_RE.search(doc):
                section = g_match.group("content")
                for item in cls._GOOGLE_ITEM_RE.finditer(section):
                    name = item.group("name")
                    first = item.group("first_line").strip()
                    cont_lines = [line.strip() for line in (item.group("cont") or "").splitlines() if line.strip()]
                    desc = " ".join([first, *cont_lines]).strip()
                    if desc and name not in params:
                        params[name] = desc

            if n_match := cls._NUMPY_SECTION_RE.search(doc):
                section = n_match.group("content")
                for item in cls._NUMPY_ITEM_RE.finditer(section):
                    name = item.group("name")
                    desc = " ".join(line.strip() for line in item.group("desc").splitlines() if line.strip())
                    if desc and name not in params:
                        params[name] = desc

        cls._CACHE[target_cls] = params
        return dict(params)


@dataclass(slots=True, frozen=True)
class ParameterSpec:
    """Introspected metadata for a single `__init__` parameter.

    Attributes:
        name (str): Parameter name.
        annotation (Any): Unwrapped type annotation.
        default (Any): Default value from `inspect.signature`, or `inspect.Parameter.empty`.
        docstring (str): Extracted help text from Google or NumPy docstrings.
        is_optional (bool): Whether `None` is an allowed value for this parameter.
    """

    name: str
    annotation: Any
    default: Any
    docstring: str = ""
    is_optional: bool = False


type WidgetBuilder = Callable[[str, ParameterSpec], TagChild]
type TypePredicate = Callable[[ParameterSpec], bool]


class WidgetSelector:
    """Introspects class parameters and maps them to PyShiny input widgets decorated with help tooltips.

    Attributes:
        default_exclude (ClassVar[set[str]]): Parameter names excluded by default during introspection.
        non_ui_types (ClassVar[tuple[type[Any], ...]]): Parameter types (such as RNG instances)
            that should be automatically omitted from generated UI forms.

    Examples:
        ```python
        from shiny_kit.widgets import WidgetSelector

        class ModelConfig:
            def __init__(self, alpha: float = 0.05, method: str = "exact") -> None:
                self.alpha = alpha
                self.method = method

        selector = WidgetSelector.default()
        widgets = selector.build_form(ModelConfig, prefix="cfg_")
        ```
    """

    default_exclude: ClassVar[set[str]] = set()
    non_ui_types: ClassVar[tuple[type[Any], ...]] = ()
    _INTROSPECT_CACHE: ClassVar[
        dict[tuple[type["WidgetSelector"], type[Any], frozenset[str]], tuple[ParameterSpec, ...]]
    ] = {}
    _DEFAULT_INSTANCES: ClassVar[dict[type["WidgetSelector"], "WidgetSelector"]] = {}

    def __init__(self) -> None:
        """Initializes `WidgetSelector` and registers default type-to-widget builder rules."""
        self._rules: list[tuple[TypePredicate, WidgetBuilder]] = []
        self._register_default_rules()

    @classmethod
    def default(cls) -> Self:
        """Returns the process-wide shared default selector instance for `cls`."""
        inst = cls._DEFAULT_INSTANCES.get(cls)
        if not isinstance(inst, cls):
            inst = cls()
            cls._DEFAULT_INSTANCES[cls] = inst
        return inst

    @classmethod
    def _is_non_ui_union_branch(cls, arg: Any) -> bool:
        """Returns True for union branches like Callable or `cls.non_ui_types` that cannot be entered as scalar UI inputs."""
        origin = get_origin(arg) or arg
        if origin in (Callable, Sequence, Mapping):
            return True
        return bool(cls.non_ui_types and isinstance(origin, type) and issubclass(origin, cls.non_ui_types))

    def register(self, predicate: TypePredicate) -> Callable[[WidgetBuilder], WidgetBuilder]:
        """Decorator to register a custom widget builder rule (evaluated first)."""

        def decorator(builder: WidgetBuilder) -> WidgetBuilder:
            self._rules.insert(0, (predicate, builder))
            return builder

        return decorator

    @staticmethod
    def _create_label(spec: ParameterSpec) -> TagChild:
        """Creates an input label with an inline tooltip icon if a docstring exists."""
        label_text = spec.name.replace("_", " ").title()
        if not spec.docstring:
            return label_text

        return shiny_ui.span(
            label_text,
            " ",
            shiny_ui.tooltip(
                shiny_ui.span(
                    shiny_ui.HTML("&#9432;"),
                    style="cursor: help; color: var(--bs-secondary, #6c757d); font-size: 0.85em;",
                ),
                spec.docstring,
                placement="right",
            ),
        )

    def _build_select_input(
        self,
        input_id: str,
        spec: ParameterSpec,
        raw_choices: Mapping[str, str],
        selected: str | None = None,
    ) -> TagChild:
        """Constructs a `shiny.ui.input_select` with optional `(None)` entry and default fallback."""
        choices: dict[str, str] = {"": "(None)"} if spec.is_optional else {}
        choices.update(raw_choices)
        if selected is None or selected not in choices:
            selected = "" if (spec.default is None and spec.is_optional) else next(iter(choices.keys()), None)
        return shiny_ui.input_select(
            input_id,
            label=self._create_label(spec),
            choices=choices,
            selected=selected,
        )

    def _register_default_rules(self) -> None:
        self._rules.extend(
            [
                (lambda s: s.annotation is bool or isinstance(s.default, bool), self._build_bool_widget),
                (lambda s: get_origin(s.annotation) is Literal, self._build_literal_widget),
                (
                    lambda s: isinstance(s.annotation, type) and issubclass(s.annotation, Enum),
                    self._build_enum_widget,
                ),
                (
                    lambda s: s.annotation is int or (isinstance(s.default, int) and not isinstance(s.default, bool)),
                    self._build_int_widget,
                ),
                (lambda s: s.annotation is float or isinstance(s.default, float), self._build_float_widget),
                (lambda _: True, self._build_text_widget),
            ]
        )

    def _build_bool_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        default_val = bool(spec.default) if spec.default not in (inspect.Parameter.empty, None) else False
        return shiny_ui.input_switch(
            input_id,
            label=self._create_label(spec),
            value=default_val,
        )

    def _build_literal_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        raw_choices = [c for c in get_args(spec.annotation) if c is not types.NoneType]
        choice_map = {str(c): str(c) for c in raw_choices}
        selected = str(spec.default) if spec.default in raw_choices else None
        return self._build_select_input(input_id, spec, choice_map, selected=selected)

    def _build_enum_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        enum_cls: type[Enum] = spec.annotation
        choice_map = {e.name: str(e.value) for e in enum_cls}
        selected = spec.default.name if isinstance(spec.default, Enum) else None
        return self._build_select_input(input_id, spec, choice_map, selected=selected)

    def _build_int_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        if spec.default is None and spec.is_optional:
            val: int | None = None
            min_val: int | None = 0
        else:
            val = spec.default if (isinstance(spec.default, int) and not isinstance(spec.default, bool)) else 0
            min_val = 0 if val >= 0 else None
        return shiny_ui.input_numeric(
            input_id,
            label=self._create_label(spec),
            value=val,  # ty: ignore[invalid-argument-type]
            min=min_val,
            step=1,
        )

    def _build_float_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        if spec.default is None and spec.is_optional:
            val: float | None = None
            step = 0.1
        else:
            val = (
                float(spec.default)
                if (isinstance(spec.default, (int, float)) and not isinstance(spec.default, bool))
                else 0.0
            )
            step = 0.01 if 0.0 < abs(val) <= 1.0 else 0.1
        return shiny_ui.input_numeric(
            input_id,
            label=self._create_label(spec),
            value=val,  # ty: ignore[invalid-argument-type]
            step=step,
        )

    def _build_text_widget(self, input_id: str, spec: ParameterSpec) -> TagChild:
        val = "" if spec.default in (inspect.Parameter.empty, None) else str(spec.default)
        return shiny_ui.input_text(
            input_id,
            label=self._create_label(spec),
            value=val,
        )

    @classmethod
    def _unwrap_annotation(cls, annotation: Any, default: Any) -> tuple[Any, bool]:
        """Unwraps Optional/Union annotations, stripping NoneType and non-UI branches like Callable."""
        is_optional = default is None
        origin = get_origin(annotation)
        if origin is types.UnionType or str(origin) == "typing.Union":
            raw_args = get_args(annotation)
            if types.NoneType in raw_args:
                is_optional = True
            non_none_args = [a for a in raw_args if a is not types.NoneType]
            ui_args = [a for a in non_none_args if not cls._is_non_ui_union_branch(a)]
            candidates = ui_args or non_none_args
            if len(candidates) == 1:
                annotation = candidates[0]
            elif candidates and all(get_origin(a) is Literal for a in candidates):
                annotation = candidates[0]
        return annotation, is_optional

    def introspect(self, target_cls: type[Any], *, exclude: set[str] | None = None) -> list[ParameterSpec]:
        """Introspects and caches `__init__` parameters, walking the MRO when `**kwargs` is forwarded to superclasses."""
        excluded = frozenset(self.default_exclude | (exclude or set()) | {"self", "args", "kwargs"})
        cache_key = (type(self), target_cls, excluded)
        cached = self._INTROSPECT_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        docs = DocstringParser.parse_parameters(target_cls)
        specs: list[ParameterSpec] = []
        seen_names: set[str] = set()

        for cls in target_cls.__mro__:
            if cls is object:
                break
            sig = inspect.signature(cls.__init__)
            has_var_keyword = False
            for name, param in sig.parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if name in excluded or name in seen_names:
                    continue
                seen_names.add(name)
                annotation, is_optional = self._unwrap_annotation(param.annotation, param.default)
                specs.append(
                    ParameterSpec(
                        name=name,
                        annotation=annotation,
                        default=param.default,
                        docstring=docs.get(name, ""),
                        is_optional=is_optional,
                    )
                )
            if not has_var_keyword:
                break

        self._INTROSPECT_CACHE[cache_key] = tuple(specs)
        return specs

    def _build_single_widget(
        self,
        spec: ParameterSpec,
        *,
        choices: Mapping[str, Sequence[str] | Mapping[str, str]] | None = None,
        prefix: str = "",
    ) -> TagChild:
        """Builds a single Shiny input widget for `spec`."""
        input_id = f"{prefix}{spec.name}" if prefix else spec.name
        if choices and spec.name in choices:
            override_choices = choices[spec.name]
            choice_map = (
                {str(k): str(v) for k, v in override_choices.items()}
                if isinstance(override_choices, Mapping)
                else {str(item): str(item) for item in override_choices}
            )
            selected = str(spec.default) if spec.default is not None else None
            return self._build_select_input(input_id, spec, choice_map, selected=selected)

        builder = next(builder for pred, builder in self._rules if pred(spec))
        return builder(input_id, spec)

    def build_form(
        self,
        target_cls: type[Any],
        *,
        exclude: set[str] | None = None,
        choices: Mapping[str, Sequence[str] | Mapping[str, str]] | None = None,
        prefix: str = "",
    ) -> list[TagChild]:
        """Builds a collection of Shiny input widgets for the target class."""
        specs = self.introspect(target_cls, exclude=exclude)
        return [self._build_single_widget(spec, choices=choices, prefix=prefix) for spec in specs]

    def extract_kwargs(
        self,
        target_cls: type[Any],
        inputs: Inputs | Mapping[str, Any],
        *,
        exclude: set[str] | None = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Extracts and casts values from Shiny's `Inputs` or a mapping to pass into `target_cls(**kwargs)`."""
        specs = self.introspect(target_cls, exclude=exclude)
        kwargs: dict[str, Any] = {}

        for spec in specs:
            input_id = f"{prefix}{spec.name}" if prefix else spec.name
            if input_id not in inputs:
                continue

            entry = inputs[input_id]
            raw_val = entry() if callable(entry) else entry

            if (raw_val is None or raw_val == "") and spec.is_optional:
                kwargs[spec.name] = None
            elif isinstance(spec.annotation, type) and issubclass(spec.annotation, Enum):
                if isinstance(raw_val, Enum):
                    kwargs[spec.name] = raw_val
                elif str(raw_val) in spec.annotation.__members__:
                    kwargs[spec.name] = spec.annotation[str(raw_val)]
                else:
                    kwargs[spec.name] = spec.annotation(raw_val)
            elif spec.annotation is bool or isinstance(spec.default, bool):
                kwargs[spec.name] = bool(raw_val)
            elif spec.annotation is int or (isinstance(spec.default, int) and not isinstance(spec.default, bool)):
                kwargs[spec.name] = int(raw_val) if raw_val is not None and raw_val != "" else spec.default
            elif spec.annotation is float or isinstance(spec.default, float):
                kwargs[spec.name] = float(raw_val) if raw_val is not None and raw_val != "" else spec.default
            else:
                kwargs[spec.name] = raw_val

        return kwargs
