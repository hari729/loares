from opti.base.bmr import bmr
from opti.base.bwr import bwr
from opti.base.bmwr import bmwr
from opti.base.mutation import random_reinit
from opti.moo.bw_selection import bw_selection
from opti.core.update import UpdateRule

BMR = UpdateRule(bw_selection, bmr, random_reinit)
BWR = UpdateRule(bw_selection, bwr, random_reinit)
BMWR = UpdateRule(bw_selection, bmwr, random_reinit)
