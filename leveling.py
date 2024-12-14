from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import numpy as np

support = np.array(list(range(1, 51)))

def evaluate(support, prev_dist, next_dist, alpha):
    score = 0
    next_nonzero = next_dist[np.isclose(next_dist, 0) == False]
    prev_nonzero = prev_dist[np.isclose(next_dist, 0) == False]
    support_nonzero = np.array(support)[np.isclose(next_dist, 0) == False]
    
    score = np.sum(next_nonzero * support_nonzero / (alpha * next_nonzero + (1 - alpha) * prev_nonzero))
    return score

prev_dist = [0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.0, 0.02976190476190476, 0.011904761904761904, 0.017857142857142856, 0.023809523809523808, 0.023809523809523808, 0.011904761904761904, 0.005952380952380952, 0.03571428571428571, 0.017857142857142856, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.023809523809523808, 0.05357142857142857, 0.023809523809523808, 0.03571428571428571, 0.02976190476190476, 0.03571428571428571, 0.047619047619047616, 0.05357142857142857, 0.03571428571428571, 0.02976190476190476, 0.023809523809523808, 0.017857142857142856, 0.03571428571428571, 0.017857142857142856, 0.02976190476190476, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.017857142857142856, 0.02976190476190476]
prev_dist = np.array(prev_dist)
alpha = 0.6
print(prev_dist, alpha)

'''def find_next_dist(support, prev_dist, alpha):
    def objective(next_dist):
        return -evaluate(support, prev_dist, next_dist, alpha)

    initial_guess = [1/len(support)] * len(prev_dist)
    bounds = [(0, 1)] * len(prev_dist)

    # Adding a linear constraint to ensure the sum of next_dist is 1
    linear_constraint = LinearConstraint([1] * len(prev_dist), [1], [1])

    result = minimize(objective, initial_guess, bounds=bounds, constraints=[linear_constraint], options={'maxiter': 1000})
    print(result)
    return result.x'''

def find_next_dist(support, prev_dist, alpha): 
    next_dist = np.random.dirichlet(np.ones(len(support)), size=1)[0]
    for _ in range(1000):
        next_dist = next_dist * np.array(support) / (alpha * next_dist + (1 - alpha) * prev_dist)
        next_dist /= next_dist.sum()

    return next_dist

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