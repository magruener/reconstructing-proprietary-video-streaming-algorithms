import itertools
from time import time
from types import FunctionType
from warnings import warn

import numpy as np
import pandas as pd
from gplearn.fitness import _Fitness, weighted_pearson, weighted_spearman, mean_square_error, root_mean_square_error, \
    log_loss, mean_absolute_error
from gplearn.functions import add2, sub2, mul2, div2, sqrt1, log1, abs1, neg1, inv1, max2, min2, sin1, cos1, tan1, \
    _Function, sig1, make_function
from gplearn.genetic import _parallel_evolve, MAX_INT, SymbolicTransformer
from gplearn.utils import _partition_estimators, check_random_state
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array

"""
Modified code directly from the gplean library (gplearn) https://gplearn.readthedocs.io/en/stable/intro.html
"""

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss}

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}


def projection_generator_function(max_arity, projection='np.mean'):
    function_list = []
    base_arity = 3
    for current_arity in range(base_arity, max_arity):
        base_str = "def experiment_file("
        for i in range(base_arity, base_arity + current_arity):
            base_str += 'x%d,' % i
        base_str = base_str[:-1]
        base_str += "):\n\treturn "
        base_str += '%s(np.vstack([' % projection
        for i in range(base_arity, base_arity + current_arity):
            base_str += 'x%d,' % i
        base_str = base_str[:-1]
        base_str += "]).T,axis = 1)"
        base_code = compile(base_str, "<string>", "exec")
        base_code = FunctionType(base_code.co_consts[0], globals(), "base_code")
        function_list.append(make_function(base_code, '%s_%d' % (projection, current_arity), arity=current_arity))
    return function_list


class GeneticFeatureGenerator(SymbolicTransformer, TransformerMixin):
    """This combines genetic feature engineering and tree based feature selection.

    A symbolic transformer is a supervised transformer that begins by building
    a population of naive random formulas to represent a relationship. The
    formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each
    successive generation of programs is then evolved from the one that came
    before it by selecting the fittest individuals from the population to
    undergo genetic operations such as crossover, mutation or reproduction.The quality of a
    population is evaluated via feature importance in this instant.
    The final population is searched for the fittest individuals with the least
    correlation to one another.

    Adapted from the GP learn library.

    Parameters
    ----------
    tree_estimator : {XGBRegressor,XGBClassifier}
    Tree estimator that is used to evaluate feature importance.

    population_size : integer, optional (default=1000)
    The number of programs in each generation.

    hall_of_fame : integer, or None, optional (default=100)
    The number of fittest programs to compare from when finding the
    least-correlated individuals for the n_components. If `None`, the
    entire final generation will be used.

    n_components : integer, or None, optional (default=10)
    The number of best programs to return after searching the hall_of_fame
    for the least-correlated individuals. If `None`, the entire
    hall_of_fame will be used.

    generations : integer, optional (default=20)
    The number of generations to evolve.

    tournament_size : integer, optional (default=20)
    The number of programs that will compete to become part of the next
    generation.

    stopping_criteria : float, optional (default=1.0)
    The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
    The range of constants to include in the formulas. If None then no
    constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
    The range of tree depths for the initial population of naive formulas.
    Individual trees will randomly choose a maximum depth from this range.
    When combined with `init_method='half and half'` this yields the well-
    known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
    - 'grow' : Nodes are chosen at random from both functions and
      terminals, allowing for smaller trees than `init_depth` allows. Tends
      to grow asymmetrical trees.
    - 'full' : Functions are chosen until the `init_depth` is reached, and
      then terminals are selected. Tends to grow 'bushy' trees.
    - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
      'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
    The functions to use when building and evolving programs. This iterable
    can include strings to indicate either individual functions as outlined
    below, or you can also include your own functions as built using the
    ``make_function`` factory from the ``functions`` module.

    Available individual functions are:

    - 'add' : addition, arity=2.
    - 'sub' : subtraction, arity=2.
    - 'mul' : multiplication, arity=2.
    - 'div' : protected division where a denominator near-zero returns 1.,
      arity=2.
    - 'sqrt' : protected square root where the absolute value of the
      argument is used, arity=1.
    - 'log' : protected log where the absolute value of the argument is
      used and a near-zero argument returns 0., arity=1.
    - 'abs' : absolute value, arity=1.
    - 'neg' : negative, arity=1.
    - 'inv' : protected inverse where a near-zero argument returns 0.,
      arity=1.
    - 'max' : maximum, arity=2.
    - 'min' : minimum, arity=2.
    - 'sin' : sine (radians), arity=1.
    - 'cos' : cosine (radians), arity=1.
    - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='pearson')
    The cc_session_identifier of the raw fitness metric. Available options include:

    - 'pearson', for Pearson's product-moment correlation coefficient.
    - 'spearman' for Spearman's rank-order correlation coefficient.

    parsimony_coefficient : float or "auto", optional (default=0.001)
    This constant penalizes large programs by adjusting their fitness to
    be less favorable for selection. Larger values penalize the program
    more which can control the phenomenon known as 'bloat'. Bloat is when
    evolution is increasing the size of programs without a significant
    increase in fitness, which is costly for computation time and makes for
    a less understandable final result. This parameter may need to be tuned
    over successive runs.

    If "auto" the parsimony coefficient is recalculated for each generation
    using c = Cov(l,experiment_file)/Var( l), where Cov(l,experiment_file) is the covariance between
    program size l and program fitness experiment_file in the population, and Var(l) is
    the variance of program sizes.

    p_crossover : float, optional (default=0.9)
    The probability of performing crossover on a tournament winner.
    Crossover takes the winner of a tournament and selects a random subtree
    from it to be replaced. A second tournament is performed to find a
    donor. The donor also has a subtree selected at random and this is
    inserted into the original parent to form an offspring in the next
    generation.

    p_subtree_mutation : float, optional (default=0.01)
    The probability of performing subtree mutation on a tournament winner.
    Subtree mutation takes the winner of a tournament and selects a random
    subtree from it to be replaced. A donor subtree is generated at random
    and this is inserted into the original parent to form an offspring in
    the next generation.

    p_hoist_mutation : float, optional (default=0.01)
    The probability of performing hoist mutation on a tournament winner.
    Hoist mutation takes the winner of a tournament and selects a random
    subtree from it. A random subtree of that subtree is then selected
    and this is 'hoisted' into the original subtrees location to form an
    offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
    The probability of performing point mutation on a tournament winner.
    Point mutation takes the winner of a tournament and selects random
    nodes from it to be replaced. Terminals are replaced by other terminals
    and functions are replaced by other functions that require the same
    number of arguments as the original node. The resulting tree forms an
    offspring in the next generation.

    Note : The above genetic operation probabilities must sum to less than
    one. The balance of probability is assigned to 'reproduction', where a
    tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
    For point mutation only, the probability that any given node will be
    mutated.

    max_samples : float, optional (default=1.0)
    The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
    Optional list of feature names, used purely for representations in
    the `print` operation or `export_graphviz`. If None, then X0, X1, etc
    will be used for representations.

    warm_start : bool, optional (default=False)
    When set to ``True``, reuse the solution of the previous call to fit
    and add more generations to the evolution, otherwise, just fit a new
    evolution.

    low_memory : bool, optional (default=False)
    When set to ``True``, only the current generation is retained. Parent
    information is discarded. For very large populations or runs with many
    generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
    The number of jobs to run in parallel for `fit`. If -1, then the number
    of jobs is set to the number of cores.

    verbose : int, optional (default=0)
    Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    Attributes
    ----------
    run_details_ : dict
    Details of the evolution process. Includes the following elements:

    - 'generation' : The generation index.
    - 'average_length' : The average program length of the generation.
    - 'average_fitness' : The average program fitness of the generation.
    - 'best_length' : The length of the best program in the generation.
    - 'best_fitness' : The fitness of the best program in the generation.
    - 'best_oob_fitness' : The out of bag fitness of the best program in
      the generation (requires `max_samples` < 1.0).
    - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicRegressor

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 tree_estimator,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div', 'max', 'min'),
                 metric='pearson',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 max_formula_length = 5,
                 p_point_replace = 0.05,
                 random_state=None,
                 time_budget_s = 100):
        super().__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)
        self.time_budget_s = time_budget_s
        self.max_formula_length = max_formula_length
        self.tree_estimator = tree_estimator
        self._best_programs = None

    def transform(self, X):
        X = super().transform(X)
        X = pd.DataFrame(X, columns=list(map(str, self._best_programs)))
        return X

    def predict(self, X):
        return self.tree_estimator.predict(self.transform(X))

    def score(self, X, y):
        return self.tree_estimator.score(self.transform(X), y)

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        fitting_start_time = time()
        try:
            self.feature_names = X.columns
        except:
            pass

        random_state = check_random_state(self.random_state)

        # Check arrays
        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)
        else:
            X, y = check_X_y(X, y, y_numeric=True)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        _, self.n_features_ = X.shape

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function cc_session_identifier %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_, len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sig1
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]
                parents = list(filter(lambda p: p.raw_fitness_ > 0, parents))
                if len(parents) < 2:
                    break

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            population = [program for program in population if program.length_ <= self.max_formula_length] # We want to impose that in order to keep it interpretable
            seen_key = set()
            population_unique = []
            for p in population :
                if str(p) not in seen_key :
                    population_unique += [p]
                seen_key.add(str(p))
            population = population_unique
            # -------------------------------------------------------------------------------
            # Modification -> fitness is now the importance score in the tree
            modified_features = np.array([p.execute(X) for p in population]).T
            modified_names = np.array([str(p) for p in population])
            modified_features = pd.DataFrame(modified_features, columns=modified_names)
            # -------------------------------------------------------------------------------
            # n_samples,n_features
            self.tree_estimator.fit(modified_features, y,sample_weight = sample_weight)
            feature_importance_fitness = self.tree_estimator.feature_importances_
            for idx in range(len(population)):
                population[idx].raw_fitness_ = feature_importance_fitness[idx]

            # --------------------------------------------------------------------------------

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(len(self._programs[old_gen - 1])):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            best_program = population[np.argmax(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            best_fitness = fitness[np.argmax(fitness)]
            if best_fitness >= self.stopping_criteria or ((time() - fitting_start_time) > self.time_budget_s):
                break

        # Find the best individuals in the final generation
        if self.hall_of_fame is not None:
            fitness = np.array(fitness)
            hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]
        else:
            self._best_programs = self._programs[-1]
        self.tree_estimator.fit(self.transform(X), y)
        return self
