import numpy as np

def get_frequencies(json):
    # Extract the data from the JSON
    data = json["response"]["data"]
    
    # Initialize a dictionary to store the frequencies
    frequencies = {}
    
    # Iterate through the data and count the frequencies
    for item in data:
        if isinstance(item[0], str):
            key = item[0]
        else:
            key = str(item[0])
        value = int(item[1])
        
        if key in frequencies:
            frequencies[key] += value
        else:
            frequencies[key] = value
    
    return frequencies

def iterate_over_days():
    import os
    import json

    all_frequencies = []
    
    # Iterate over all files in the /data directory
    for filename in os.listdir('data/'):
        if filename.endswith('.json'):
            filepath = os.path.join('data/', filename)
            
            # Open and load the JSON file
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                
                # Get frequencies from the JSON data
                frequencies = get_frequencies(json_data)

                # Append the frequencies to the all_frequencies list
                all_frequencies.append(frequencies)

    return all_frequencies

def log_prob(observed_distribution, expected_distribution):
    return np.sum(observed_distribution * np.log(1e-18 + expected_distribution)) - np.sum(observed_distribution * np.log(1e-18 + observed_distribution / sum(observed_distribution)))

# Call the function to iterate over days and get the frequencies
all_frequencies = iterate_over_days()
support = list(map(int,all_frequencies[0].keys()))
support = np.array(support, dtype=np.float64)
data = [np.array(list(freq.values()), dtype=np.float64) for freq in all_frequencies]
aggregate_data = np.sum(data[1:], axis=0)
aggregate_distribution = aggregate_data / np.sum(aggregate_data)
data = [dist / sum(dist) for dist in data]
optimal = np.array(list(i + 1 for i in range(len(aggregate_distribution)))) / sum(i + 1 for i in range(len(aggregate_distribution)))
support = np.array(list(range(1, 51)))

human_data_raw = [102,77,83,89,87,69,150,84,111,51,87,71,65,59,67,74,64,70,53,79,50,91,58,56,87,52,62,50,53,57,63,61,77,47,58,85,69,45,52,59,55,60,65,86,70,61,50,64,81,116]
human_data_scaled = np.array(human_data_raw, dtype=np.float64) * support
human_distribution = human_data_scaled / sum(human_data_scaled)
prev_dist = human_distribution

def find_next_dist(support, prev_dist, alpha): 
    next_dist = np.random.dirichlet(np.ones(len(support)), size=1)[0]
    for _ in range(50):
        denom = alpha * next_dist + (1 - alpha) * prev_dist
        if np.isclose(np.min(denom), 0, atol=1e-18):
            next_dist = np.array([(i if np.isclose(d, 0) else 0) for i, d in zip(support, denom)])
            return next_dist / sum(next_dist)
        else:
            next_dist = next_dist * np.array(support) / denom
            next_dist /= next_dist.sum()
    return next_dist

print("OPTIMAL:", log_prob(aggregate_data, optimal))
print("HUMAN:", log_prob(aggregate_data, human_distribution))

import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

depth = 25

def objective(params):
    init_dist = params
    alpha, beta = 0.7, 0.7
    distributions = [init_dist]
    for i in range(1,depth+1):
        next_dist = find_next_dist(support, distributions[-1], alpha)
        distributions.append(next_dist * beta**i)

    agg = np.sum(distributions, axis=0)
    agg /= np.sum(agg)
    res = -log_prob(aggregate_data, agg)
    # print(alpha, beta, res)

    return res

from scipy.special import softmax

def monte_carlo_hill_climbing(objective, initial_params, iterations=1000, step_size=1.0):
    best_params = initial_params
    best_score = objective(best_params)
    
    for _ in range(iterations):
        directions = [np.concatenate((np.random.uniform(-0.5, 0.5, size=len(initial_params)), np.random.uniform(-1, 1, size=0))) * step_size / np.sqrt(_ + 2) for __ in range(10)]
        candidate_params = [best_params + direction for direction in directions]
        candidate_score = np.zeros(10)
        for i in range(10):
            probs = softmax(candidate_params[i])
            candidate_score[i] = objective(probs)

        best_candidate = np.argmin(candidate_score)
        best_score = candidate_score[best_candidate]
        best_params = candidate_params[best_candidate]

        while 1:
            direction = directions[best_candidate]
            candidate_params = best_params + direction
            candidate_score = objective(candidate_params)
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = candidate_params
            else:
                break
        print("Iteration ", _, "Best Score: ", best_score, "Best Params: ", best_params)
    return best_params, best_score

# Initial parameters (random initialization)
initial_params = np.log(human_distribution)

initial_params = np.array([-3.52992134, -4.60665201, -4.88328221, -5.55384425, -5.35237503, -5.88174295,
 -6.75155967, -4.97345642, -5.73249808, -4.78313707, -4.57356283, -4.06723307,
 -5.17423059, -4.38089257, -5.15743483, -7.71908637, -5.06353784, -5.23593638,
 -5.94619331, -5.06115998, -6.61883463, -5.38008689, -2.66850182, -5.50029203,
 -2.89846485, -5.31316644, -2.85596213, -2.73621327, -2.85239025, -3.15084559,
 -6.03297386, -2.38515051, -2.63885808, -2.39091915, -2.47107902, -3.22360169,
 -2.00265615, -2.12236337, -3.7449739, -2.7833307, -2.44098504, -2.62410244,
 -2.31750855, -3.72569116, -2.42709067, -3.15158354, -2.0789759, -4.12225313,
 -2.42367446, -1.83976004])

# Run the Monte Carlo Hill Climbing algorithm
best_params, best_score = monte_carlo_hill_climbing(objective, initial_params)

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Extract the initial distribution, alpha, and beta from the best parameters
init_dist = best_params[:-1]
alpha, beta = best_params[-1:]

