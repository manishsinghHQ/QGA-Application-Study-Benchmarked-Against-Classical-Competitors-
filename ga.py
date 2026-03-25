import numpy as np
from fitness import evaluate_solution

def ga(X, y, pop_size=20, generations=30):
    n_features = X.shape[1]

    population = np.random.randint(2, size=(pop_size, n_features))

    best_fitness = -1
    best_solution = None

    convergence = []

    for gen in range(generations):
        fitness_vals = np.array([evaluate_solution(ind, X, y) for ind in population])

        best_idx = np.argmax(fitness_vals)

        if fitness_vals[best_idx] > best_fitness:
            best_fitness = fitness_vals[best_idx]
            best_solution = population[best_idx]

        new_pop = []

        for _ in range(pop_size):
            parents = population[np.random.choice(pop_size, 2)]
            crossover_point = np.random.randint(n_features)

            child = np.concatenate([
                parents[0][:crossover_point],
                parents[1][crossover_point:]
            ])

            mutation = np.random.rand(n_features) < 0.1
            child = np.logical_xor(child, mutation).astype(int)

            new_pop.append(child)

        population = np.array(new_pop)
        convergence.append(best_fitness)

    return best_fitness, convergence, best_solution
