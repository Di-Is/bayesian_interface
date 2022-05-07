import typing
from dataclasses import dataclass

from bayesian_interface.data_structure.builder import SaveType, DataFactory, T
import bayesian_interface.data_structure.structure_parts as parts


@dataclass
class AutoCorrResult:

    method_name: parts.AbsAttr
    steps: parts.AbsArray
    iats: parts.AbsArray


class AutoCorrResultFactory(DataFactory):
    @classmethod
    def get_dataclass(cls) -> typing.Type[T]:
        return AutoCorrResult

    @classmethod
    def set_attr_kwargs_default(
        cls, save_type: SaveType, kwargs: dict[str, dict[str, typing.Any]]
    ) -> dict:
        return kwargs

    @classmethod
    def set_array_kwargs_default(
        cls, save_type: SaveType, kwargs: dict[str, dict[str, typing.Any]]
    ) -> dict:
        match save_type:
            case SaveType.netcdf:
                # data dimension name
                if "dimensions" not in kwargs["iats"]:
                    kwargs["iats"] |= {"dimensions": ("draw", "dim")}
                if "dimensions" not in kwargs["steps"]:
                    kwargs["steps"] |= {"dimensions": ("draw",)}
            case save_type if save_type in [SaveType.hdf5, SaveType.netcdf]:
                # compression
                if "compression" not in kwargs["iats"]:
                    kwargs["iats"] |= {"compression": "gzip"}
                if "compression_opts" not in kwargs["steps"]:
                    kwargs["steps"] |= {"compression_opts": 9}

        return kwargs
