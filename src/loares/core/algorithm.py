from loares.core.flow import FlowHandler
from loares.core.update import UpdateRule
from loares.core.population import PopulationHandler
from loares.core.problem import ProblemHandler


class Algorithm:
    def __init__(self, name, update_function,
                 selection_function, mutation,
                 sorting_function, mods=None, flowhandler=FlowHandler):
        self.name = name
        self.update_function = update_function
        self.selection_function = selection_function
        self.mutation = mutation
        self.sorting_function = sorting_function
        self.mods = mods if mods is not None else []
        self.flowhandler = flowhandler

    def __call__(self, problem):
        return self.flowhandler(
            ProblemHandler(problem),
            UpdateRule(self.selection_function, self.update_function, self.mutation),
            PopulationHandler(self.sorting_function),
            self.mods,
        )

    def get_info(self):
        return {
            "name": self.name,
            "BaseFunction": self.update_function.__name__.upper(),
            "Mutation": self.mutation.__name__,
            "Selection": self.selection_function.__name__,
            "Sorting": self.sorting_function.__name__,
            "mods": [mod.__name__ for mod in self.mods],
        }
