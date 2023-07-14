# (c) EPFL - LTS5, 2023

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from dataclasses import dataclass
from types import NoneType

import pytest

from coco_api.validation import (AndConstraintSet,
                                 ConstraintSetValidationError,
                                 ConstraintValidationError,
                                 DataclassConstraint, InstanceConstraint,
                                 NoneConstraint, OrConstraintSet)


def test_none_constraint():
	NoneConstraint(value=True).validate(None)
	NoneConstraint(value=False).validate(1)
	NoneConstraint(value=True, obj=None).validate()
	NoneConstraint(value=False, obj=1).validate()
	with pytest.raises(ConstraintValidationError):
		NoneConstraint(value=False).validate(None)
	with pytest.raises(ConstraintValidationError):
		NoneConstraint(value=True).validate(1)
	with pytest.raises(ConstraintValidationError):
		NoneConstraint(value=False, obj=None).validate()
	with pytest.raises(ConstraintValidationError):
		NoneConstraint(value=True, obj=1).validate()


def test_instance_constraint():
	InstanceConstraint(int, value=True).validate(1)
	InstanceConstraint(int, value=False).validate(None)
	InstanceConstraint(int, value=True, obj=1).validate()
	InstanceConstraint(int, value=False, obj=None).validate()
	with pytest.raises(ConstraintValidationError):
		InstanceConstraint(int, value=False).validate(1)
	with pytest.raises(ConstraintValidationError):
		InstanceConstraint(int, value=True).validate(None)
	with pytest.raises(ConstraintValidationError):
		InstanceConstraint(int, value=False, obj=1).validate()
	with pytest.raises(ConstraintValidationError):
		InstanceConstraint(int, value=True, obj=None).validate()


def test_dataclass_constraint():
	class C: pass
	@dataclass
	class DC: pass
	DataclassConstraint(value=False).validate(C())
	DataclassConstraint(value=True).validate(DC())
	DataclassConstraint(value=False, obj=C()).validate()
	DataclassConstraint(value=True, obj=DC()).validate()
	with pytest.raises(ConstraintValidationError):
		DataclassConstraint(value=True).validate(C())
	with pytest.raises(ConstraintValidationError):
		DataclassConstraint(value=False).validate(DC())
	with pytest.raises(ConstraintValidationError):
		DataclassConstraint(value=True, obj=C()).validate()
	with pytest.raises(ConstraintValidationError):
		DataclassConstraint(value=False, obj=DC()).validate()


def test_or_constraint_set():
	with pytest.raises(ValueError):
		OrConstraintSet()
	OrConstraintSet(NoneConstraint(obj=None)).validate()
	OrConstraintSet(NoneConstraint(obj=None), NoneConstraint(obj=1)).validate()
	OrConstraintSet(NoneConstraint(obj=1), NoneConstraint(obj=None)).validate()
	OrConstraintSet(NoneConstraint(value=False), InstanceConstraint(int)).validate(1)
	with pytest.raises(ConstraintSetValidationError):
		OrConstraintSet(NoneConstraint(obj=1)).validate()
	with pytest.raises(ConstraintSetValidationError):
		OrConstraintSet(NoneConstraint(value=False, obj=None), NoneConstraint(obj=1)).validate()
	with pytest.raises(ConstraintSetValidationError):
		OrConstraintSet(NoneConstraint(value=True), InstanceConstraint(str)).validate(1)


def test_and_constraint_set():
	with pytest.raises(ValueError):
		AndConstraintSet()
	AndConstraintSet(NoneConstraint(obj=None)).validate()
	AndConstraintSet(NoneConstraint(obj=None), InstanceConstraint(int, obj=1)).validate()
	AndConstraintSet(NoneConstraint(), InstanceConstraint(NoneType)).validate(None)
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(NoneConstraint(obj=1)).validate()
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(NoneConstraint(obj=None), InstanceConstraint(int, obj=None)).validate()
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(NoneConstraint(), InstanceConstraint(int)).validate(None)


def test_compound_constraint_set():
	with pytest.raises(ValueError):
		AndConstraintSet(OrConstraintSet())
	AndConstraintSet(OrConstraintSet(NoneConstraint(obj=None), NoneConstraint(obj=1))).validate()
	AndConstraintSet(OrConstraintSet(NoneConstraint(obj=None), NoneConstraint(obj=1)), NoneConstraint(obj=None)).validate()
	AndConstraintSet(
		OrConstraintSet(NoneConstraint(obj=None), NoneConstraint(obj=1)),
		OrConstraintSet(NoneConstraint(obj=None), NoneConstraint(obj=1))
		).validate()
	AndConstraintSet(OrConstraintSet(NoneConstraint())).validate(None)
	AndConstraintSet(OrConstraintSet(NoneConstraint()), NoneConstraint()).validate(None)
	AndConstraintSet(OrConstraintSet(NoneConstraint()), OrConstraintSet(NoneConstraint())).validate(None)
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(OrConstraintSet(NoneConstraint(obj=1), NoneConstraint(obj=1))).validate()
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(OrConstraintSet(NoneConstraint(obj=1), NoneConstraint(obj=1)), NoneConstraint(obj=1)).validate()
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(
			OrConstraintSet(NoneConstraint(obj=1), NoneConstraint(obj=1)),
			OrConstraintSet(NoneConstraint(obj=1), NoneConstraint(obj=1))
			).validate()
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(OrConstraintSet(NoneConstraint())).validate(1)
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(OrConstraintSet(NoneConstraint()), NoneConstraint()).validate(1)
	with pytest.raises(ConstraintSetValidationError):
		AndConstraintSet(OrConstraintSet(NoneConstraint()), OrConstraintSet(NoneConstraint())).validate(1)
