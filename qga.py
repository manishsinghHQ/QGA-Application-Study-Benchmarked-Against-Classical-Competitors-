import numpy as np
from fitness import evaluate_solution

def qga(X, y, pop_size=20, generations=30):
    n_features = X.shape[1]

    theta = np.random.uniform(0, np.pi/2, (pop_size, n_features))

    best_solution = None
    best_fitness = -1

    convergence = []

    for gen in range(generations):
        population = (np.random.rand(pop_size, n_features) < np.sin(theta)**2).astype(int)

        fitness_vals = []

        for i in range(pop_size):
            fit = evaluate_solution(population[i], X, y)
            fitness_vals.append(fit)

            if fit > best_fitness:
                best_fitness = fit
                best_solution = population[i]

        for i in range(pop_size):
            for j in range(n_features):
                if population[i][j] != best_solution[j]:
                    theta[i][j] += 0.05

        convergence.append(best_fitness)

    return best_fitness, convergence, best_solution
