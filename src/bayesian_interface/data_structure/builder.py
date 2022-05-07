import typing
import enum
from dataclasses import is_dataclass, fields
from abc import ABCMeta, abstractmethod
from bayesian_interface.data_structure.misc import dynamically_load_class

import bayesian_interface.data_structure.structure_parts as parts


T = typing.TypeVar("T")


class SaveType(enum.Enum):
    memory = 0
    hdf5 = 1
    netcdf = 2


class DataFactory(metaclass=ABCMeta):
    @classmethod
    def create(
        cls,
        save_type: SaveType = SaveType.memory,
        attr_kwargs: typing.Optional[dict] = None,
        array_kwargs: typing.Optional[dict] = None,
    ) -> T:
        data_cls = cls.get_dataclass()
        attr_kwargs = cls.format_kwargs(data_cls, parts.AbsAttr, attr_kwargs)
        attr_kwargs = cls.set_attr_kwargs_default(save_type, attr_kwargs)
        array_kwargs = cls.format_kwargs(data_cls, parts.AbsArray, array_kwargs)
        array_kwargs = cls.set_array_kwargs_default(save_type, array_kwargs)

        save_type_name = str(save_type).replace("SaveType.", "")
        module = f"bayesian_interface.data_structure.{save_type_name}"
        attr_factory = dynamically_load_class(module, "AttrFactory")
        array_factory = dynamically_load_class(module, "ArrayFactory")

        kwargs = {}

        for var_info in fields(data_cls):
            name = var_info.name
            if var_info.type is parts.AbsAttr:
                kwargs[name] = attr_factory.create(name=name, **attr_kwargs[name])
            elif var_info.type is parts.AbsArray:
                kwargs[name] = array_factory.create(name=name, **array_kwargs[name])
            else:
                raise TypeError(
                    f"The annotation {var_info.type} of variable {name} is an invalid."
                )

        return data_cls(**kwargs)

    @classmethod
    @abstractmethod
    def set_attr_kwargs_default(cls, save_type: SaveType, kwargs: dict) -> dict:
        pass

    @classmethod
    @abstractmethod
    def set_array_kwargs_default(cls, save_type: SaveType, kwargs: dict) -> dict:
        pass

    @classmethod
    @abstractmethod
    def get_dataclass(cls) -> typing.Type[T]:
        pass

    @staticmethod
    def format_kwargs(
        data_class: object.__class__, check_class: object.__class__, kwargs: dict
    ) -> dict:
        from copy import deepcopy

        kwargs = deepcopy(kwargs)
        names = [
            key for key, val in data_class.__annotations__.items() if val is check_class
        ]
        # None対応
        if kwargs is None:
            kwargs = {name: {} for name in names}
        else:
            for name in names:
                kwargs[name] = kwargs[name] if name in kwargs else {}
        return kwargs
