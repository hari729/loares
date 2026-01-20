from opti.algorithms.moo.base import MO_BMR, MO_BWR, MO_BMWR
from opti.algorithms.moo.archive import MO_BMR_ARCHIVE, MO_BWR_ARCHIVE, MO_BMWR_ARCHIVE
from opti.algorithms.moo.samp import MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP
from opti.algorithms.moo.opposition import MO_BMR_OPPOSITION, MO_BWR_OPPOSITION, MO_BMWR_OPPOSITION

from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD

algolist = [MO_BMR, MO_BWR, MO_BMWR,
            MO_BMR_ARCHIVE, MO_BWR_ARCHIVE, MO_BMWR_ARCHIVE,
            MO_BMR_OPPOSITION, MO_BWR_OPPOSITION, MO_BMWR_OPPOSITION,
            MO_BMR_SAMP, MO_BWR_SAMP, MO_BMWR_SAMP]

pymoo_algolist = [NSGA2, NSGA3, MOPSO_CD, CMOPSO]
