import numpy as np
from fitness import evaluate_solution

def pso(X, y, pop_size=20, generations=30):
    n_features = X.shape[1]

    particles = np.random.rand(pop_size, n_features)
    velocities = np.random.rand(pop_size, n_features)

    pbest = particles.copy()
    pbest_val = np.array([
        evaluate_solution((p > 0.5).astype(int), X, y)
        for p in particles
    ])

    gbest_idx = np.argmax(pbest_val)
    gbest = pbest[gbest_idx]
    gbest_val = pbest_val[gbest_idx]

    convergence = []

    for _ in range(generations):
        for i in range(pop_size):
            velocities[i] = 0.5 * velocities[i] + \
                            1.5 * np.random.rand() * (pbest[i] - particles[i]) + \
                            1.5 * np.random.rand() * (gbest - particles[i])

            particles[i] = particles[i] + velocities[i]

            solution = (particles[i] > 0.5).astype(int)
            fit = evaluate_solution(solution, X, y)

            if fit > pbest_val[i]:
                pbest[i] = particles[i]
                pbest_val[i] = fit

        gbest_idx = np.argmax(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]

        convergence.append(gbest_val)

    best_solution = (gbest > 0.5).astype(int)

    return gbest_val, convergence, best_solution
