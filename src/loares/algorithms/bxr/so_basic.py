from loares.core.algorithm import Algorithm
from loares.operators.sorting import bw_sorting
from loares.operators.bxr import *
from loares.operators.selection import bw_selection
from loares.operators.mutation import random_reinit


SO_BMR = Algorithm("BMR", bmr, bw_selection, random_reinit, bw_sorting)
SO_BWR = Algorithm("BWR", bwr, bw_selection, random_reinit, bw_sorting)
SO_BMWR = Algorithm("BMWR", bmwr, bw_selection, random_reinit, bw_sorting)
