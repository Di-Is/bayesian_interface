import typing
from enum import Enum
from dataclasses import dataclass

from bayesian_interface.data_structure.builder import DataFactory, SaveType
import bayesian_interface.data_structure.structure_parts as parts


class CriteriaType(Enum):
    equal = 0
    upper = 1
    lower = 2


@dataclass
class ConvergenceResult:
    # Criterion method name
    criterion_method: parts.AbsAttr
    # convergence flag
    convergence: parts.AbsAttr
    # The number of converge step
    convergence_step: parts.AbsAttr
    # The Threshold convergence criterion
    threshold: parts.AbsAttr
    # Threshold Type
    threshold_type: parts.AbsAttr  # upper or lower equal
    # Number of Chain ID for check convergence
    chain_ids: parts.AbsAttr

    # The checked step
    steps: parts.AbsArray
    # The checked criterion values corresponding steps
    criterion_values: parts.AbsArray
    convergences: parts.AbsArray
    convergence_steps: parts.AbsArray


class ConvergenceResultFactory(DataFactory):
    @classmethod
    def set_array_kwargs_default(cls, save_type: SaveType, kwargs: dict) -> dict:
        return kwargs

    @classmethod
    def set_attr_kwargs_default(cls, save_type: SaveType, kwargs: dict) -> dict:
        return kwargs

    @classmethod
    def get_dataclass(cls) -> typing.Type[ConvergenceResult]:
        return ConvergenceResult
