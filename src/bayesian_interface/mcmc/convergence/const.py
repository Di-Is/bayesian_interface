from enum import Enum
from .min_afactor import MinAfactorStrategy
from .max_archange import MaxArchangeStrategy
from .gr import GR
from .gr_rank import GRRank
from .max_archange import MaxArchangeStrategy


IAT_NEED_METHODS = (
    MinAfactorStrategy.algorithm_name,
    MaxArchangeStrategy.algorithm_name,
)


class Method(Enum):
    manual = 0
    gelman_rubin = 1
    gelman_rubin_rank = 2
    min_afactor = 3
    max_archange = 4
