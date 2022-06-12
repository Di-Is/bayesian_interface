import typing
from abc import ABCMeta, abstractmethod

from dataclasses import fields, dataclass

import bayesian_interface.data_structure as bay_data


T = typing.TypeVar("T")


class AbsDefaultParam(metaclass=ABCMeta):
    @classmethod
    def get_param(cls, save_type: bay_data.SaveType) -> dataclass:
        match save_type:
            case bay_data.SaveType.memory:
                par_default = cls.get_memory()
            case bay_data.SaveType.hdf5:
                par_default = cls.get_hdf5()
            case bay_data.SaveType.netcdf:
                par_default = cls.get_netcdf4()
            case _:
                raise ValueError(f"{save_type} is invalid save type.")
        return par_default

    @classmethod
    @abstractmethod
    def get_memory(cls) -> dict:
        pass

    @classmethod
    @abstractmethod
    def get_hdf5(cls) -> dict:
        pass

    @classmethod
    @abstractmethod
    def get_netcdf4(cls) -> dict:
        pass


class AbsDataFactory(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_dataclass(cls) -> dataclass.__class__:  # noqa
        pass

    @classmethod
    @abstractmethod
    def get_dflt_param(cls) -> AbsDefaultParam.__class__:
        pass

    @classmethod
    def create(
        cls,
        save_type: bay_data.SaveType = bay_data.SaveType.memory,
        each: typing.Optional[dict[str, dict]] = None,
        **common,
    ) -> T:

        kwargs = {}
        data_cls = cls.get_dataclass()
        default_param = cls.get_dflt_param().get_param(save_type)
        each = cls.merge_param(default_param, each, **common)

        for var_info in fields(data_cls):
            name = var_info.name
            save_type_name = str(save_type).replace("SaveType.", "")
            module = f"bayesian_interface.data_structure.{save_type_name}"
            if var_info.type is bay_data.AbsAttr:
                factory = bay_data.dynamically_load_class(module, "AttrFactory")
            elif var_info.type is bay_data.AbsArray:
                factory = bay_data.dynamically_load_class(module, "ArrayFactory")
            else:
                raise TypeError(
                    f"The annotation {var_info.type} of variable {name} is an invalid."
                )
            kwargs[name] = factory.create(name=name, **each[name])

        return data_cls(**kwargs)

    @classmethod
    def merge_param(
        cls,
        default_param: dict,
        each: typing.Optional[dict[str, dict]] = None,
        **common,
    ) -> dict:
        common = {} if common is None else common.copy()
        each = {} if each is None else each.copy()

        names = list(default_param.keys())
        result = {}
        for name in names:
            v = default_param[name]
            if name in each:
                each[name] = v | each[name]
            else:
                each[name] = v.copy()
            result[name] = common | each[name]

        return result
