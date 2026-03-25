import numpy as np
from fitness import evaluate_solution

def de(X, y, pop_size=20, generations=30):
    n_features = X.shape[1]

    pop = np.random.rand(pop_size, n_features)

    best_solution = None
    best_fitness = -1

    convergence = []

    for gen in range(generations):
        new_pop = []

        for i in range(pop_size):
            a, b, c = pop[np.random.choice(pop_size, 3, replace=False)]
            mutant = np.clip(a + 0.5 * (b - c), 0, 1)

            cross = np.random.rand(n_features) < 0.7
            trial = np.where(cross, mutant, pop[i])

            trial_bin = (trial > 0.5).astype(int)
            target_bin = (pop[i] > 0.5).astype(int)

            trial_fit = evaluate_solution(trial_bin, X, y)
            target_fit = evaluate_solution(target_bin, X, y)

            if trial_fit > target_fit:
                new_pop.append(trial)

                if trial_fit > best_fitness:
                    best_fitness = trial_fit
                    best_solution = trial_bin
            else:
                new_pop.append(pop[i])

                if target_fit > best_fitness:
                    best_fitness = target_fit
                    best_solution = target_bin

        pop = np.array(new_pop)
        convergence.append(best_fitness)

    return best_fitness, convergence, best_solution
