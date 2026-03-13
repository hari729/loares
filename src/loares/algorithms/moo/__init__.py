from loares.algorithms.moo.base import MO_BMR, MO_BWR, MO_BMWR
from loares.algorithms.moo.archive import (
    MO_BMR_Archive,
    MO_BWR_Archive,
    MO_BMWR_Archive,
)
from loares.algorithms.moo.samp import MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP
from loares.algorithms.moo.opposition import (
    MO_BMR_Opposition,
    MO_BWR_Opposition,
    MO_BMWR_Opposition,
)

from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD

algolist = [
    MO_BMR,
    MO_BWR,
    MO_BMWR,
    MO_BMR_Archive,
    MO_BWR_Archive,
    MO_BMWR_Archive,
    MO_BMR_Opposition,
    MO_BWR_Opposition,
    MO_BMWR_Opposition,
    MO_BMR_SAMP,
    MO_BWR_SAMP,
    MO_BMWR_SAMP,
]

pymoo_algolist = [NSGA2, NSGA3, MOPSO_CD, CMOPSO]
