from loares.core.algorithm import Algorithm
from loares.operators.sorting import ranking_crowding
from loares.operators.bxr import *
from loares.operators.selection import random_bw_selection
from loares.operators.mutation import random_reinit
from loares.operators.mods import local_search


MO_BMR = Algorithm("MO-BMR", bmr, random_bw_selection, random_reinit, ranking_crowding, [local_search])
MO_BWR = Algorithm("MO-BWR", bwr, random_bw_selection, random_reinit, ranking_crowding, [local_search])
MO_BMWR = Algorithm("MO-BMWR", bmwr, random_bw_selection, random_reinit, ranking_crowding, [local_search])


