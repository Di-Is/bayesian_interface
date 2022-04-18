from dataclasses import asdict

import bayesian_interface.data_structure.structure_parts as parts
import bayesian_interface.data_structure.memory as memory
import bayesian_interface.data_structure.hdf5 as hdf5


class Director:
    """データ構造を生成するディレクター"""

    def __init__(self):
        self.builder = Builder()

    def build(
        self,
        cls: parts.DataStructureBase.__class__,
        attr_params: list,
        array_params: list,
    ) -> parts.DataStructureBase:
        attr_kwargs = {}
        for p_attr in attr_params:
            attr_kwargs[p_attr.name] = self.builder.build_attribute(p_attr)
        array_kwargs = {}
        for p_arr in array_params:
            array_kwargs[p_arr.name] = self.builder.build_array(p_arr)
        res = cls(**attr_kwargs, **array_kwargs)
        return res


def get_attr_cls(name) -> parts.AttrBase.__class__:
    dct = {memory.AttrParam: memory.Attr, hdf5.AttrParam: hdf5.Attr}
    return dct[name]


def get_array_cls(name) -> parts.ArrayBase.__class__:
    dct = {memory.ArrayParam: memory.Array, hdf5.ArrayParam: hdf5.Array}
    return dct[name]


class Builder:
    """データ構造を生成するビルダー"""

    @staticmethod
    def build_attribute(attr_param: parts.AttrParamBase):
        cls = get_attr_cls(attr_param.__class__)
        return cls(**asdict(attr_param))

    @staticmethod
    def build_array(array_param: parts.ArrayParamBase):
        cls = get_array_cls(array_param.__class__)
        return cls(**asdict(array_param))
