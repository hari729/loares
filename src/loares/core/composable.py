"""
ModularAlgorithm — a composable evolutionary algorithm framework built on pymoo.

This provides two classes:

RecombinationVariant(InfillCriterion)
    Wraps pool_selection + recombination + mutation into pymoo's infill contract.
    Sits inside InfillCriterion.do()'s retry loop, so repair and dedup apply
    automatically after each batch.

ModularAlgorithm(Algorithm)
    The main loop orchestrator. Composes:
        - Any InfillCriterion (RecombinationVariant, Mating, DE Variant, etc.)
        - A list of Mods (additive infill sources)
        - Any Survival
        - Repair applied to all outputs (core + mods)

    This sits alongside GeneticAlgorithm as a peer inheriting from Algorithm,
    not as a replacement. GeneticAlgorithm hardcodes Selection → Crossover →
    Mutation. ModularAlgorithm is agnostic — it delegates to whatever
    InfillCriterion it receives and adds the mods concept on top.
"""

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.infill import InfillCriterion
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.duplicate import NoDuplicateElimination, DefaultDuplicateElimination
from pymoo.util.misc import has_feasible
from pymoo.util.optimum import filter_optimum


class RecombinationVariant(InfillCriterion):
    """
    InfillCriterion that uses population-level recombination.

    This is the counterpart to pymoo's Mating — while Mating does
    Selection → Crossover → Mutation on parent pairs, RecombinationVariant
    does PoolSelection → Recombination → Mutation on the whole population.

    It plugs into InfillCriterion.do()'s retry loop, which means
    repair and eliminate_duplicates (passed to __init__) run automatically
    after each call to _do().

    Parameters
    ----------
    pool_selection : PoolSelection
        Constructs the parent pool (best, worst, mean, random arrays).
    recombination : Recombination
        The update equation (BMR, BWR, BMWR, or custom).
    mutation : pymoo Mutation or None
        Applied after recombination. RandomReinit, PM, or None.
    repair : pymoo Repair or None
        Fixes invalid X values (bounds, rounding, etc.).
        Runs inside the retry loop after each batch.
    eliminate_duplicates : DuplicateElimination or None
        Removes duplicates. Runs inside the retry loop.
    n_max_iterations : int
        Max retries in the InfillCriterion retry loop.
    """

    def __init__(self,
                 pool_selection,
                 recombination,
                 mutation=None,
                 repair=None,
                 eliminate_duplicates=None,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__(
            repair=repair,
            eliminate_duplicates=eliminate_duplicates,
            n_max_iterations=n_max_iterations,
            **kwargs
        )

        self.pool_selection = pool_selection
        self.recombination = recombination
        self.mutation = mutation

        # Validate that the selection provides what the recombination needs
        provided = self.pool_selection.provides
        required = self.recombination.requires
        if provided and required:
            missing = required - provided
            if missing:
                raise ValueError(
                    f"{self.recombination.__class__.__name__} requires pool keys "
                    f"{required}, but {self.pool_selection.__class__.__name__} "
                    f"only provides {provided}. Missing: {missing}"
                )

    def _do(self, problem, pop, n_offsprings,
            algorithm=None, random_state=None, **kwargs):

        # Step 1: Build the parent pool
        pool = self.pool_selection.do(
            pop, algorithm=algorithm, random_state=random_state
        )

        # Step 2: Apply the recombination equation
        X = self.recombination.do(
            problem, pop, pool, random_state=random_state
        )

        # Step 3: Create offspring population
        off = Population.new("X", X[:n_offsprings])

        # Step 4: Apply mutation if provided
        if self.mutation is not None:
            off = self.mutation.do(problem, off, random_state=random_state)

        return off


class ModularAlgorithm(Algorithm):
    """
    A composable evolutionary algorithm.

    Every component is a swappable module:
        - sampling: how to create the initial population
        - infill: how to generate core offspring (any InfillCriterion)
        - survival: how to select survivors from merged pool
        - mods: list of additive infill sources
        - repair: bounds/feasibility repair (applied to all offspring)

    The generation cycle is:
        1. _infill():
            a. Core offspring via self.infill_criterion.do() [with internal retry/repair/dedup]
            b. For each mod: generate extra candidates
            c. Repair all outputs
        2. Algorithm.next() evaluates the combined offspring
        3. _advance():
            a. Merge current pop + evaluated offspring
            b. Survival selects pop_size survivors

    Parameters
    ----------
    pop_size : int
        Population size.
    sampling : pymoo Sampling or np.ndarray or Population
        Initial population generation strategy.
    infill : InfillCriterion
        Core offspring generator. Can be:
        - RecombinationVariant(pool_selection, recombination, mutation)
        - pymoo Mating(selection, crossover, mutation)
        - pymoo DE Variant
        - Any custom InfillCriterion
    survival : pymoo Survival
        Survivor selection strategy (RankAndCrowding, FitnessSurvival, etc.)
    n_offsprings : int or None
        Number of core offspring per generation. Defaults to pop_size.
    mods : list of Mod or None
        Additive infill sources. Each generates extra candidates.
    repair : pymoo Repair or None
        Applied to mod outputs and during initialization.
        Core offspring repair is handled by the InfillCriterion internally.
    eliminate_duplicates : DuplicateElimination or bool or None
        Applied during initialization. Core offspring dedup is handled
        by the InfillCriterion internally.
    advance_after_initial_infill : bool
        Whether to run survival on the initial population.
        True for NSGA-II style (sets rank/crowding on initial pop).
    """

    def __init__(self,
                 pop_size,
                 sampling,
                 infill,
                 survival,
                 n_offsprings=None,
                 mods=None,
                 repair=None,
                 eliminate_duplicates=None,
                 advance_after_initial_infill=True,
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.n_offsprings = n_offsprings if n_offsprings is not None else pop_size
        self.survival = survival
        self.mods = mods if mods is not None else []
        self.advance_after_initial_infill = advance_after_initial_infill

        # Repair for initialization and mod outputs
        self.repair = repair if repair is not None else NoRepair()

        # Duplicate elimination for initialization
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = (
                eliminate_duplicates
                if eliminate_duplicates is not None
                else NoDuplicateElimination()
            )

        # Initialization uses sampling + repair + dedup
        self.initialization = Initialization(
            sampling,
            repair=self.repair,
            eliminate_duplicates=self.eliminate_duplicates,
        )

        # The core infill criterion (RecombinationVariant, Mating, etc.)
        self.infill_criterion = infill

    def _initialize_infill(self):
        """Generate the initial population via sampling."""
        return self.initialization.do(
            self.problem, self.pop_size,
            algorithm=self, random_state=self.random_state
        )

    def _initialize_advance(self, infills=None, **kwargs):
        """
        Optionally run survival on the initial population.

        When advance_after_initial_infill is True, survival processes
        the initial population to set metadata (rank, crowding distance)
        that the selection needs in the first real generation.
        """
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(
                self.problem, infills,
                n_survive=len(infills),
                algorithm=self,
                random_state=self.random_state,
            )

    def _infill(self):
        """
        Generate offspring: core InfillCriterion + mods.

        The InfillCriterion handles its own repair/dedup internally
        (via the retry loop in InfillCriterion.do()).

        Mods generate additional candidates independently.
        Repair is applied to everything at the end.
        """
        # Core offspring
        off = self.infill_criterion.do(
            self.problem, self.pop, self.n_offsprings,
            algorithm=self, random_state=self.random_state
        )

        if off is None or len(off) == 0:
            self.termination.force_termination = True
            return None

        # Mods: each independently generates extra individuals
        for mod in self.mods:
            extra = mod.do(
                self.problem, self.pop,
                algorithm=self, random_state=self.random_state
            )
            if extra is not None and len(extra) > 0:
                off = Population.merge(off, extra)

        # Repair all outputs (harmless on already-repaired core offspring)
        off = self.repair.do(self.problem, off, random_state=self.random_state)

        return off

    def _advance(self, infills=None, **kwargs):
        """
        Merge current population with evaluated offspring, apply survival.
        """
        pop = self.pop

        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(
            self.problem, pop,
            n_survive=self.pop_size,
            algorithm=self,
            random_state=self.random_state,
        )

    def _set_optimum(self):
        """
        Set the current optimum from the population.

        For multi-objective: non-dominated front (rank 0).
        For single-objective: best by filter_optimum.
        """
        if self.problem.n_obj > 1:
            rank = self.pop.get("rank")
            if rank is not None and np.any(rank == 0):
                self.opt = self.pop[rank == 0]
            else:
                self.opt = filter_optimum(self.pop, least_infeasible=True)
        else:
            self.opt = filter_optimum(self.pop, least_infeasible=True)
