from loares.core.algorithm import Algorithm
from loares.operators.sorting import ranking_crowding
from loares.operators.bxr import *
from loares.operators.selection import random_bw_selection
from loares.operators.mutation import random_reinit
from loares.operators.mods import local_search, opposition


MO_BMR_O = Algorithm("MO-BMR-Opposition", bmr, random_bw_selection, random_reinit, ranking_crowding,
                   [local_search, opposition])
MO_BWR_O = Algorithm("MO-BWR-Opposition", bwr, random_bw_selection, random_reinit, ranking_crowding,
                   [local_search, opposition])
MO_BMWR_O = Algorithm("MO-BMWR-Opposition", bmwr, random_bw_selection, random_reinit, ranking_crowding,
                    [local_search, opposition])
