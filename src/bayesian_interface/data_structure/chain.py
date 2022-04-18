from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

import bayesian_interface.data_structure.builder as builder
import bayesian_interface.data_structure.structure_parts as parts
import bayesian_interface.data_structure.hdf5 as hdf5
import bayesian_interface.data_structure.memory as memory


@dataclass
class SamplingResult:
    """データ構造のみを定義"""

    # データ構造を定義,　初期化はビルダーパターンを利用し初期化する
    chain_id: parts.AttrBase
    sampler_name: parts.AttrBase
    chain: parts.ArrayBase
    lnprob: parts.ArrayBase


class SamplingResultMaker:

    attrs = {"chain_id": {}, "sampler_name": {}}
    arrs = {
        "chain": {"dtype": np.float64},
        "lnprob": {"dtype": np.float64},
    }

    def get(
        self, mode: str = "memory", fpath: Optional[str] = None, gpath: str = "mcmc"
    ) -> SamplingResult:
        if mode == "memory":
            attrs = [
                memory.AttrParam(name, **kwargs) for name, kwargs in self.attrs.items()
            ]
            arrs = [
                memory.ArrayParam(name, **kwargs) for name, kwargs in self.arrs.items()
            ]
        else:
            if fpath is None:
                raise ValueError
            h5_kwargs = {"fpath": fpath, "gpath": gpath}
            attrs = [
                hdf5.AttrParam(name, **kwargs, **h5_kwargs)
                for name, kwargs in self.attrs.items()
            ]
            arrs = [
                hdf5.ArrayParam(name, **kwargs, **h5_kwargs)
                for name, kwargs in self.arrs.items()
            ]
        return builder.Director().build(SamplingResult, attrs, arrs)


if __name__ == "__main__":
    chain = SamplingResultMaker().get()
    print(chain)
