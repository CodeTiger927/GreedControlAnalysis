from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import numpy as np

support = list(range(1, 51))

def evaluate(support, prev_dist, next_dist, alpha):
    score = 0
    for j, prev, next in zip(support, prev_dist, next_dist):
        if not np.isclose(next, 0):
            score += next * j / (alpha * next + (1 - alpha) * prev)
    return score

prev_dist = np.random.dirichlet(np.ones(len(support)), size=1)[0]
alpha = 0.4
print(prev_dist, alpha)

def find_next_dist(support, prev_dist, alpha):
    def objective(next_dist):
        return -evaluate(support, prev_dist, next_dist, alpha)

    initial_guess = [1/len(support)] * len(prev_dist)
    bounds = [(0, 1)] * len(prev_dist)

    # Adding a linear constraint to ensure the sum of next_dist is 1
    linear_constraint = LinearConstraint([1] * len(prev_dist), [1], [1])

    result = minimize(objective, initial_guess, bounds=bounds, constraints=[linear_constraint])
    return result.x

def combine_dist(support, prev_dist, next_dist, alpha):
    return [alpha * next + (1 - alpha) * prev for prev, next in zip(prev_dist, next_dist)]

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.plot(support, prev_dist, label=f'Iteration 0', marker='o')
plt.xlabel('Support')
plt.ylabel('Distribution')
plt.legend()
for it in tqdm(range(1,10)):
    next_dist = find_next_dist(support, prev_dist, alpha)
    prev_dist = next_dist
    
    plt.plot(support, prev_dist, label=f'Iteration {it}', marker='o')
    plt.xlabel('Support')
    plt.ylabel('Distribution')
    plt.legend()
plt.show()