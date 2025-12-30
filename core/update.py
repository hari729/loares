
def null_mutator(problem, new_gen):
    return new_gen

class UpdateRule():
    def __init__(self, selection, base_fucntion, mutation, mods):
        self.selection = selection
        self.base_fucntion = base_fucntion
        self.mutation = mutation
        if mutation is None:
            self.mutation = null_mutator
        else:
            self.mutation = mutation

    def next_gen(self, problem, population):
        new_gen = self.basefunction(problem, population, self.selection(population))
        new_gen = self.mutation(problem, new_gen)

        for mod in self.pmods:
            new_gen = np.vstack([new_gen,mod(problem, population)])

        new_gen = problem.variable_modifier(new_gen)

        return new_gen
