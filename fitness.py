import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def evaluate_solution(solution, X, y):
    if np.sum(solution) == 0:
        return 0

    selected_features = X[:, solution == 1]

    X_train, X_test, y_train, y_test = train_test_split(
        selected_features, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    penalty = np.sum(solution) / len(solution)

    return acc - 0.1 * penalty
