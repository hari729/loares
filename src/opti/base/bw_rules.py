from opti.base.bmr import bmr
from opti.base.bwr import bwr
from opti.base.bmwr import bmwr
from opti.base.mutation import random_reinit
from opti.core.update import UpdateRule
from opti.moo.bw_selection import random_bw_selection as bw_selection

# def bw_selection(population):
#     best = population.solutions[0,:]
#     worst = population.solutions[-1,:]
#     return {"best":best, "worst":worst}

BMR = UpdateRule(bw_selection, bmr, random_reinit)
BWR = UpdateRule(bw_selection, bwr, random_reinit)
BMWR = UpdateRule(bw_selection, bmwr, random_reinit)
