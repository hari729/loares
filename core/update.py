
def null_mutator(problem, new_gen):
    return new_gen

class UpdateRule():
    def __init__(self, selection, base_function, mutation, mods):
        self.selection = selection
        self.base_function = base_function
        self.mutation = mutation
        if mutation is None:
            self.mutation = null_mutator
        else:
            self.mutation = mutation
        self.pmods = mods

    def next_gen(self, problem, population):
        new_gen = self.base_function(problem, population, self.selection(population))
        new_gen = self.mutation(problem, new_gen)

        for mod in self.pmods:
            new_gen = np.vstack([new_gen,mod(problem, population)])

        new_gen = problem.variable_modifier(new_gen)

        return new_gen
