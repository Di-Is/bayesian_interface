from enum import Enum
from .emcee_sampler import EmceeEnsemble
from .zeus_sampler import ZeusStrategy
from .sample_adaptive_sampler import SampleAdaptiveStrategy


class SamplerType(Enum):
    NORMAL = 0
    ENSEMBLE = 1


ENSEMBLE_SAMPLER = (EmceeEnsemble.method_name, ZeusStrategy.method_name)
NORMAL_SAMPLER = (SampleAdaptiveStrategy.method_name,)
SAMPLER = NORMAL_SAMPLER + ENSEMBLE_SAMPLER
