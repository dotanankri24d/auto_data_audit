"""Signature-derived parameters and explicit, local-code extension discovery."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints
from ..contracts import ValidationError

INFRA = {"embedding": 1, "distance": 2, "method": 1, "policy": 1}
ENTRY = {"embeddings": ("embedding", "fit"), "distances": ("distance", "distance"),
         "methods": ("method", "analyze"), "policies": ("policy", "decide")}


@dataclass(frozen=True)
class Parameter:
    name: str
    annotation: Any
    default: Any
    required: bool
    nullable: bool
    choices: tuple = ()
    kind: type = str


def parameter_schema(function, skip=0, supplied=None):
    """Build editable fields, rejecting ambiguous signatures explicitly."""
    supplied = supplied or {}
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = {}
    fields = []
    for p in list(signature.parameters.values())[skip:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY):
            raise ValidationError(f"{p.name}: open/positional-only signatures are unsupported")
        if p.name in supplied:
            continue
        required = p.default is inspect.Parameter.empty
        annotation = hints.get(p.name, p.annotation)
        if annotation is inspect.Parameter.empty:
            if required or p.default is None:
                raise ValidationError(f"{p.name}: an explicit type is required")
            annotation = type(p.default)
        nullable = False
        if get_origin(annotation) in (Union, types.UnionType):
            args = get_args(annotation)
            nullable = type(None) in args
            rest = [a for a in args if a is not type(None)]
            if len(rest) != 1 or not nullable:
                raise ValidationError(f"{p.name}: unsupported union {annotation}")
            annotation = rest[0]
        choices = get_args(annotation) if get_origin(annotation) is Literal else ()
        kind = type(choices[0]) if choices else annotation
        if kind not in (bool, int, float, str) or any(type(c) is not kind for c in choices):
            raise ValidationError(f"{p.name}: unsupported UI type {annotation}; supply a registered component or value")
        fields.append(Parameter(p.name, annotation, None if required else p.default,
                                required, nullable, choices, kind))
    return fields


def validate_parameters(function, skip, params):
    sig = inspect.signature(function)
    try:
        bound = sig.bind(*([object()] * skip), **params)
    except TypeError as exc:
        raise ValidationError(str(exc)) from exc
    bound.apply_defaults()
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = {}
    for p in list(sig.parameters.values())[skip:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ValidationError(f"{p.name}: open signatures are unsupported")
        v = bound.arguments[p.name]
        t = hints.get(p.name, p.annotation)
        if t is inspect.Parameter.empty:
            t = type(p.default) if p.default is not inspect.Parameter.empty and p.default is not None else Any
        if not _matches(v, t):
            raise ValidationError(f"{p.name}: expected {t}, got {type(v).__name__}")
    return {p.name: bound.arguments[p.name] for p in list(sig.parameters.values())[skip:]}


def _matches(value, annotation):
    if annotation is Any or isinstance(annotation, str):
        return True
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_matches(value, a) for a in get_args(annotation))
    if origin is Literal:
        return value in get_args(annotation) and any(type(value) is type(a) for a in get_args(annotation))
    if annotation in (bool, int, str):
        return type(value) is annotation
    if annotation is float:
        return type(value) in (float, int)
    if annotation is type(None):
        return value is None
    return True  # arbitrary Python objects remain usable in the code API


@dataclass(frozen=True)
class BoundComponent:
    kind: str
    function: Any
    params: dict
    name: str
    source_hash: str | None = None

    def __call__(self, *args):
        return self.function(*args, **self.params)

    def fit(self, X):
        return self(X)

    def analyze(self, data):
        return self(data)

    def decide(self, samples):
        return self(samples)

    def describe(self):
        try:
            source = inspect.getsource(self.function)
        except (OSError, TypeError):
            source = repr(self.function)
        return {"component": self.name, "kind": self.kind, "parameters": self.params,
                "source_hash": self.source_hash or hashlib.sha256(source.encode()).hexdigest(),
                "module": self.function.__module__}


def bind(kind, function, name=None, **params):
    values = validate_parameters(function, INFRA[kind], params)
    return BoundComponent(kind, function, values, name or function.__name__)


def bind_embedding(function, **params):
    return bind("embedding", function, **params)


def bind_distance(function, **params):
    return bind("distance", function, **params)


def bind_method(function, *, name=None, **params):
    return bind("method", function, name=name, **params)


def bind_policy(function, **params):
    return bind("policy", function, **params)


@dataclass(frozen=True)
class DiscoveredComponent:
    kind: str
    name: str
    function: Any
    path: Path
    source_hash: str
    fields: list

    def bind(self, **params):
        c = bind(self.kind, self.function, name=params.pop("name", self.name), **params)
        return BoundComponent(c.kind, c.function, c.params, c.name, self.source_hash)


def discover_plugins(directory):
    """Execute only explicitly selected trusted Python files. Never call for uploaded data."""
    components, errors = [], {}
    root = Path(directory).expanduser()
    if not root.is_dir():
        return [], {str(root): "Extension directory does not exist"}
    for folder, (kind, entry) in ENTRY.items():
        for path in sorted((root / folder).glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                source = path.read_bytes()
                digest = hashlib.sha256(source).hexdigest()
                namespace = {"__name__": f"data_audit_extension_{digest}", "__file__": str(path)}
                exec(compile(source, str(path), "exec"), namespace)
                fn = namespace.get(entry)
                if not inspect.isfunction(fn) or fn.__module__ != namespace["__name__"]:
                    raise ValidationError(f"Define a local function named {entry}")
                fields = parameter_schema(fn, INFRA[kind])
                components.append(DiscoveredComponent(kind, path.stem, fn, path, digest, fields))
            except Exception as exc:
                errors[str(path)] = f"{type(exc).__name__}: {exc}"
    return components, errors
