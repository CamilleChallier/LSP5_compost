# (c) EPFL - LTS5, 2023

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


class _ConstraintValidationError(Exception, ABC):
	@property
	@abstractmethod
	def short_msg(self):
		...


class ConstraintValidationError(_ConstraintValidationError):
	cnstr: "Constraint"
	obj: Any
	
	@property
	def short_msg(self):
		return f"{self.cnstr} on {self.obj} of type {type(self.obj)}"
	
	def __init__(self, cnstr: "Constraint", obj: Any):
		self.cnstr = cnstr
		self.obj = obj
		super().__init__(self.short_msg)


@dataclass
class ConstraintSetValidationError(_ConstraintValidationError):
	cnstr_set: "ConstraintSet"
	ex_list: list["_ConstraintValidationError"]
	
	@property
	def short_msg(self):
		msg_hdr = f"{self.cnstr_set}:"
		msg_body = '\n'.join(f"  - {ex.short_msg}" for ex in self.ex_list)
		msg = f"{msg_hdr}\n{msg_body}"
		return msg
	
	def __init__(self, cnstr_set: "ConstraintSet", ex_list: list["_ConstraintValidationError"]):
		self.cnstr_set = cnstr_set
		self.ex_list = ex_list
		err_msg = f"Validation error for constraint set {cnstr_set} due to unsatisfied constraints:\n{self.short_msg}"
		super().__init__(err_msg)


class ConstraintBase(ABC):
	@abstractmethod
	def validate(self, *args, **kwargs):
		...


class _UNDEFINED:
	pass


@dataclass(kw_only=True)
class Constraint(ConstraintBase):
	value: bool = True
	obj: Any = field(default=_UNDEFINED, repr=False)
	
	@abstractmethod
	def _get_value(self, obj: Any) -> bool:
		...
	
	def _validate(self, obj: Any):
		if self._get_value(obj) != self.value:
			raise ConstraintValidationError(self, obj)
	
	def validate(self, obj: Any = _UNDEFINED):
		assert (obj is _UNDEFINED) ^ (self.obj is _UNDEFINED)  # (obj is defined) xor (self.obj is defined)
		if obj is not _UNDEFINED:
			self._validate(obj)
		else:
			self._validate(self.obj)


@dataclass
class NoneConstraint(Constraint):
	def _get_value(self, obj: Any) -> bool:
		return obj is None


@dataclass
class DataclassConstraint(Constraint):
	def _get_value(self, obj: Any) -> bool:
		return dataclasses.is_dataclass(obj)


@dataclass
class InstanceConstraint(Constraint):
	_type: type
	
	def _get_value(self, obj: Any) -> bool:
		return isinstance(obj, self._type)


class ConstraintSet(ConstraintBase):
	def __repr__(self) -> str:
		return f"<{self.__class__.__name__}>"
	
	@property
	@abstractmethod
	def agg_fn(self) -> Callable[[bool, bool], bool]:
		...
	
	cs: tuple[ConstraintBase]
	
	def __init__(self, *cs: ConstraintBase):
		if len(cs) == 0:
			raise ValueError(f"{self.__class__.__name__} requires at least one constraint (0 given)")
		self.cs = cs
	
	def _validate(self, cnstr: ConstraintBase, obj: Any) -> None | Exception:
		try:
			cnstr.validate(obj)
		except _ConstraintValidationError as ex:
			return ex
		else:
			return None
	
	def validate(self, obj: Any = _UNDEFINED):
		val_res_list = [self._validate(cnstr, obj) for cnstr in self.cs]
		if not self.agg_fn(val_res is None for val_res in val_res_list):
			ex_list = list(filter(lambda x: isinstance(x, _ConstraintValidationError), val_res_list))
			raise ConstraintSetValidationError(self, ex_list)


class OrConstraintSet(ConstraintSet):
	agg_fn = any


class AndConstraintSet(ConstraintSet):
	agg_fn = all
