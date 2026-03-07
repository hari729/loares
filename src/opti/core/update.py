def null_mutator(problem, new_gen):
    return new_gen


class UpdateRule:
    def __init__(self, selection, base_function, mutation):
        self.selection = selection
        self.base_function = base_function
        if mutation is None:
            self.mutation = null_mutator
        else:
            self.mutation = mutation

    def next_gen(self, problem, population):
        new_gen = self.base_function(problem, population, self.selection(population))
        new_gen = self.mutation(problem, new_gen)
        return new_gen

    def get_info(self):
        dictionary = {
            "name": str(self.__class__.__name__).replace("_", "-"),
            "BaseFunction": str(self.base_function.__name__.upper()),
            "Mutation": str(self.mutation.__name__),
            "Selection": str(self.selection.__name__),
        }
        return dictionary
