"""Shared pydantic base for hpcom's immutable models.

Every public configuration and result object in hpcom derives from
:class:`HpcomModel`. The base fixes four behaviours that plain
``pydantic.BaseModel`` gets wrong for a numerical library:

* **Unknown keyword arguments are errors** (``extra="forbid"``). A typo such
  as ``modulaton_order=16`` raises instead of being silently dropped.
* **Array fields are read-only.** Fields annotated with one of the
  ``*ArrayField`` / ``*ArrayView`` types below are coerced to a fixed dtype
  and stored with ``writeable=False``, so a frozen model cannot be mutated
  through its arrays. ``*ArrayField`` copies (small configuration data);
  ``*ArrayView`` stores a read-only *view* (large signal data: no copy, and
  the caller's own array stays writeable).
* **Equality and hashing understand arrays.** ``==`` compares array fields
  with :func:`numpy.array_equal` instead of raising
  ``ValueError: truth value of an array is ambiguous``.
* **Copies are validated.** :meth:`HpcomModel.replace` (and ``model_copy``
  with ``update=``, which is routed to it) rebuilds the model through its
  validators, so constraints are re-checked and nothing can go stale.

Models round-trip through JSON: ``model_dump(mode="json")`` encodes arrays as
lists (complex arrays as ``{"real": [...], "imag": [...]}``), and fields
annotated ``Polymorphic[Base]`` carry a ``"type"`` tag naming the concrete
class, so e.g. a whole :class:`~hpcom.simulation.Simulation` can be stored
next to its results and rebuilt with ``model_validate``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, ClassVar, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    WithJsonSchema,
    WrapSerializer,
)
from pydantic_core import PydanticSerializationError

__all__ = [
    "BoolArrayView",
    "ComplexArrayField",
    "ComplexArrayView",
    "FloatArrayField",
    "HpcomModel",
    "IndexArrayField",
    "IndexArrayView",
    "Polymorphic",
    "UInt8ArrayField",
    "UInt8ArrayView",
    "readonly",
]

_M = TypeVar("_M", bound="HpcomModel")
_T = TypeVar("_T")

_TYPE_KEY = "type"


# ---------------------------------------------------------------------------
# Read-only array field types
# ---------------------------------------------------------------------------


def _decode_json_array(value: Any) -> Any:
    """Accept the JSON encoding produced by :func:`_encode_array`."""
    if isinstance(value, Mapping) and set(value) == {"real", "imag"}:
        return np.asarray(value["real"], dtype=np.float64) + 1j * np.asarray(
            value["imag"], dtype=np.float64
        )
    return value


def _encode_array(value: np.ndarray) -> Any:
    if np.iscomplexobj(value):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()}
    return value.tolist()


def _coercer(dtype: type[np.generic], *, copy: bool) -> Callable[[Any], np.ndarray]:
    target = np.dtype(dtype)

    def coerce(value: Any) -> np.ndarray:
        value = _decode_json_array(value)
        src = np.asarray(value)
        if target.kind != "c" and np.iscomplexobj(src):
            raise ValueError(f"expected a real-valued {target.name} array, got complex values")
        if (
            target.kind in "iub"
            and src.dtype.kind not in "iub"
            and src.size
            and (not np.all(np.isfinite(src)) or np.any(src != np.round(src)))
        ):
            raise ValueError(f"expected integer values for a {target.name} array")
        arr = np.array(src, dtype=target, copy=True) if copy else np.asarray(src, dtype=target)
        if not copy:
            arr = arr.view()  # freezing a view leaves the caller's array writeable
        arr.flags.writeable = False
        return arr

    return coerce


_ARRAY_SERIALIZER = PlainSerializer(_encode_array, when_used="json")


def _schema(name: str) -> WithJsonSchema:
    return WithJsonSchema({"type": "array", "description": f"numpy {name} array"})


#: Read-only complex128 array, copied on construction (configuration data).
ComplexArrayField: TypeAlias = Annotated[
    npt.NDArray[np.complex128],
    BeforeValidator(_coercer(np.complex128, copy=True)),
    _ARRAY_SERIALIZER,
    _schema("complex128"),
]
#: Read-only float64 array, copied on construction.
FloatArrayField: TypeAlias = Annotated[
    npt.NDArray[np.float64],
    BeforeValidator(_coercer(np.float64, copy=True)),
    _ARRAY_SERIALIZER,
    _schema("float64"),
]
#: Read-only uint8 array, copied on construction.
UInt8ArrayField: TypeAlias = Annotated[
    npt.NDArray[np.uint8],
    BeforeValidator(_coercer(np.uint8, copy=True)),
    _ARRAY_SERIALIZER,
    _schema("uint8"),
]
#: Read-only intp array, copied on construction.
IndexArrayField: TypeAlias = Annotated[
    npt.NDArray[np.intp],
    BeforeValidator(_coercer(np.intp, copy=True)),
    _ARRAY_SERIALIZER,
    _schema("intp"),
]
#: Read-only complex128 *view* (no copy) — for large signal arrays.
ComplexArrayView: TypeAlias = Annotated[
    npt.NDArray[np.complex128],
    BeforeValidator(_coercer(np.complex128, copy=False)),
    _ARRAY_SERIALIZER,
    _schema("complex128"),
]
#: Read-only uint8 view (no copy) — for bit streams.
UInt8ArrayView: TypeAlias = Annotated[
    npt.NDArray[np.uint8],
    BeforeValidator(_coercer(np.uint8, copy=False)),
    _ARRAY_SERIALIZER,
    _schema("uint8"),
]
#: Read-only intp view (no copy) — for decision indices.
IndexArrayView: TypeAlias = Annotated[
    npt.NDArray[np.intp],
    BeforeValidator(_coercer(np.intp, copy=False)),
    _ARRAY_SERIALIZER,
    _schema("intp"),
]
#: Read-only bool view (no copy) — for masks.
BoolArrayView: TypeAlias = Annotated[
    npt.NDArray[np.bool_],
    BeforeValidator(_coercer(np.bool_, copy=False)),
    _ARRAY_SERIALIZER,
    _schema("bool"),
]


def readonly(array: npt.ArrayLike, dtype: npt.DTypeLike | None = None) -> np.ndarray:
    """Return a read-only view of ``array`` (converted to ``dtype`` if given)."""
    out: np.ndarray = np.asarray(array, dtype=dtype).view()
    out.flags.writeable = False
    return out


# ---------------------------------------------------------------------------
# Polymorphic fields
# ---------------------------------------------------------------------------


def _class_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


_REPR_KEY = "repr"


def _resolve_polymorphic(value: Any) -> Any:
    """Turn ``{"type": "pkg.module.Class", ...}`` back into an instance.

    Only classes that are already imported (hence registered) are resolved;
    loading JSON never triggers an import, so an untrusted config cannot run
    arbitrary modules.
    """
    if isinstance(value, Mapping) and _TYPE_KEY in value:
        key = value[_TYPE_KEY]
        if set(value) == {_TYPE_KEY, _REPR_KEY}:
            raise ValueError(
                f"{key!r} was recorded only by its repr ({value[_REPR_KEY]!r}) because it is "
                "not an HpcomModel; it cannot be restored from JSON"
            )
        cls = HpcomModel._registry.get(key)
        if cls is None:
            raise ValueError(
                f"unknown model type {key!r}; import the module that defines it before loading"
            )
        return cls.model_validate({k: v for k, v in value.items() if k != _TYPE_KEY})
    return value


def _tag_polymorphic(
    value: Any, handler: SerializerFunctionWrapHandler, info: SerializationInfo
) -> Any:
    if isinstance(value, HpcomModel):
        # Dump with the *runtime* class so subclass fields are kept.
        data = value.model_dump(mode=info.mode)
        return {_TYPE_KEY: _class_key(type(value)), **data}
    if info.mode == "json":
        try:
            return handler(value)
        except PydanticSerializationError:
            # A duck-typed component (e.g. a user's link element that is not an
            # HpcomModel): record what it was rather than fail the whole dump.
            return {_TYPE_KEY: _class_key(type(value)), _REPR_KEY: repr(value)}
    return handler(value)


#: ``Polymorphic[Base]`` — a field holding any subclass of ``Base``
#: (``Transmitter``, ``Receiver``, link elements …). The runtime subclass is
#: kept on ``model_dump`` and restored on ``model_validate``.
Polymorphic: TypeAlias = Annotated[
    _T,
    BeforeValidator(_resolve_polymorphic),
    WrapSerializer(_tag_polymorphic, when_used="always"),
]


# ---------------------------------------------------------------------------
# Equality helpers
# ---------------------------------------------------------------------------


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_values_equal, a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class HpcomModel(BaseModel):
    """Immutable, validated, array-aware pydantic model (see module docstring)."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_default=True,
        # inf/nan (e.g. the SNR of a noiseless run) survive a JSON round trip.
        ser_json_inf_nan="constants",
    )

    #: Qualified class name → class, for restoring ``Polymorphic`` fields.
    _registry: ClassVar[dict[str, type[HpcomModel]]] = {}

    #: Groups of mutually exclusive fields. When :meth:`replace` sets one
    #: member of a group, the other members are reset to their defaults.
    _exclusive_fields: ClassVar[tuple[tuple[str, ...], ...]] = ()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        HpcomModel._registry[_class_key(cls)] = cls

    # ---- copying -------------------------------------------------------

    def replace(self: _M, **changes: Any) -> _M:
        """Return a copy with ``changes`` applied, re-running all validation."""
        fields = type(self).model_fields
        unknown = sorted(set(changes) - set(fields))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} has no field(s) {unknown}; valid fields: {sorted(fields)}"
            )
        values = {name: getattr(self, name) for name in fields}
        for group in type(self)._exclusive_fields:
            if any(name in changes for name in group):
                for name in group:
                    if name not in changes:
                        values[name] = fields[name].get_default(call_default_factory=True)
        values.update(changes)
        return type(self).model_validate(values)

    def model_copy(self: _M, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> _M:
        """Copy the model; with ``update`` the result is fully re-validated."""
        if update:
            return self.replace(**dict(update))
        return super().model_copy(deep=deep)

    # ---- comparison ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return all(
            _values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
        )

    def __hash__(self) -> int:
        return hash(
            (type(self), tuple(_hashable(getattr(self, n)) for n in type(self).model_fields))
        )
