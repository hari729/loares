import numpy as np
from pymoo.core.operator import Operator


class Mods(Operator):
    def __init__(
        self, operator_list=None, function_list=None, name=None, vtype=None, repair=None
    ) -> None:
        super().__init__(name, vtype, repair)
        self.operators = operator_list if operator_list else []
        self.functions = function_list if function_list else []

    def _do(self, problem, elem, *args, random_state, **kwargs):
        mod_results = []
        for operator in self.operators:
            mod_results.append(
                operator.do(problem, elem, *args, random_state=random_state, **kwargs)
            )

        for function in self.functions:
            mod_results.append(
                function(problem, elem, *args, random_state=random_state, **kwargs)
            )

        result = np.concatenate(mod_results)
        return result
