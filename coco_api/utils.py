# (c) EPFL - LTS5, 2023

import dataclasses
import types
import typing
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .validation import AndConstraintSet, InstanceConstraint, NoneConstraint, OrConstraintSet


@runtime_checkable
@dataclass
class Dataclass(Protocol):
    pass


def indexing_tree_expander(cls: Optional[type] = None, *, index_key: str | typing.Hashable = None):
	def expand_index_subtree(_type: type, node: Any) -> Any:
		InstanceConstraint(typing._SpecialForm, value=False).validate(_type)  # Any or equivalent generic (e.g. Union / Optional without args)
		if dataclasses.is_dataclass(_type):
			InstanceConstraint(dict).validate(node)
			dc_inst = _type(**node)
			propagate_expansion(dc_inst)
			return dc_inst
		elif hasattr(_type, "__origin__"):  # it's a generic
			generic_type_orig = _type.__origin__
			if generic_type_orig == list:
				assert len(_type.__args__) == 1, "Generic 'list' type hint must have one argument type"
				InstanceConstraint(list).validate(node)
				generic_type_arg = _type.__args__[0]
				for idx in range(len(node)):
					node[idx] = expand_index_subtree(generic_type_arg, node[idx])
				return node
			elif generic_type_orig == dict:
				assert len(_type.__args__) == 2, "Generic 'dict' type hint must have two argument types"
				OrConstraintSet(
					InstanceConstraint(dict, obj=node),
					AndConstraintSet(
						InstanceConstraint(list, obj=node),
						NoneConstraint(obj=index_key, value=False)
					)
				).validate()
				generic_type_arg = _type.__args__[1]
				if isinstance(node, dict):
					for key in node:
						node[key] = expand_index_subtree(generic_type_arg, node[key])
					return node
				else:  # list
					# Index list elements by attribute
					index_node = {}
					for el in node:
						el = expand_index_subtree(generic_type_arg, el)
						assert isinstance(el, dict) or dataclasses.is_dataclass(el)
						if isinstance(el, dict):
							key = el[index_key]
						else:
							key = getattr(el, index_key)
						index_node[key] = el
					return index_node
			else:
				# e.g. Union / Optional with args
				if generic_type_orig in [typing.Union, types.UnionType]:
					ex_list = []
					for generic_type_arg in _type.__args__:
						try:
							return expand_index_subtree(generic_type_arg, node)
						except Exception as ex:
							ex_list.append(ex)
					raise Exception(ex_list)  # TODO raise better exception?
		else:
			InstanceConstraint(_type).validate(node)
			return node
	def propagate_expansion(dc: Dataclass):
		for field in dataclasses.fields(dc):
			field_val = getattr(dc, field.name)
			field_val = expand_index_subtree(field.type, field_val)
			setattr(dc, field.name, field_val)
	def decorator(cls):
		_post_init = cls.__post_init__ if hasattr(cls, "__post_init__") else None
		def __post_init__(self):
			if _post_init is not None:
				_post_init(cls)
			propagate_expansion(self)
		setattr(cls, __post_init__.__name__, __post_init__)
		return cls
	if cls is None:
		return decorator
	else:
		return decorator(cls)
