import typing
from dataclasses import dataclass

from bayesian_interface.data_structure.builder import SaveType, DataFactory, T
import bayesian_interface.data_structure.structure_parts as parts


@dataclass
class SamplingResult:

    chain_id: parts.AbsAttr
    sampler_name: parts.AbsAttr
    chain: parts.AbsArray
    lnprob: parts.AbsArray


class SamplingResultFactory(DataFactory):
    @classmethod
    def get_dataclass(cls) -> typing.Type[T]:
        return SamplingResult

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
                if "dimensions" not in kwargs["chain"]:
                    kwargs["chain"] |= {"dimensions": ("draw", "walker", "dim")}
                if "dimensions" not in kwargs["lnprob"]:
                    kwargs["lnprob"] |= {"dimensions": ("draw", "walker")}
            case save_type if save_type in [SaveType.hdf5, SaveType.netcdf]:
                # compression
                if "compression" not in kwargs["chain"]:
                    kwargs["chain"] |= {"compression": "gzip"}
                if "compression_opts" not in kwargs["chain"]:
                    kwargs["chain"] |= {"compression_opts": 9}
                if "compression" not in kwargs["lnprob"]:
                    kwargs["lnprob"] |= {"compression": "gzip"}
                if "compression_opts" not in kwargs["lnprob"]:
                    kwargs["lnprob"] |= {"compression_opts": 9}
        return kwargs
