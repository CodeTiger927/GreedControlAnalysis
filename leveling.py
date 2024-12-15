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
    return np.sum(observed_distribution * np.log(1e-18 + expected_distribution))

# Call the function to iterate over days and get the frequencies
all_frequencies = iterate_over_days()
support = list(map(int,all_frequencies[0].keys()))
support = np.array(support, dtype=np.float64)
data = [np.array(list(freq.values()), dtype=np.float64) for freq in all_frequencies]
aggregate_data = np.sum(data[1:], axis=0)
aggregate_distribution = aggregate_data / np.sum(aggregate_data)
data = [dist / sum(dist) for dist in data]
optimal = np.array(list(i + 1 for i in range(len(aggregate_distribution)))) / sum(i + 1 for i in range(len(aggregate_distribution)))


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

human_data_raw = [102,77,83,89,87,69,150,84,111,51,87,71,65,59,67,74,64,70,53,79,50,91,58,56,87,52,62,50,53,57,63,61,77,47,58,85,69,45,52,59,55,60,65,86,70,61,50,64,81,116]
human_data_scaled = np.array(human_data_raw, dtype=np.float64) * support
human_distribution = human_data_scaled / sum(human_data_scaled)

prev_dist = human_distribution
# prev_dist = [0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.0, 0.011904761904761904, 0.0, 0.0, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.0, 0.02976190476190476, 0.011904761904761904, 0.017857142857142856, 0.023809523809523808, 0.023809523809523808, 0.011904761904761904, 0.005952380952380952, 0.03571428571428571, 0.017857142857142856, 0.011904761904761904, 0.017857142857142856, 0.011904761904761904, 0.023809523809523808, 0.05357142857142857, 0.023809523809523808, 0.03571428571428571, 0.02976190476190476, 0.03571428571428571, 0.047619047619047616, 0.05357142857142857, 0.03571428571428571, 0.02976190476190476, 0.023809523809523808, 0.017857142857142856, 0.03571428571428571, 0.017857142857142856, 0.02976190476190476, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.017857142857142856, 0.02976190476190476]
# prev_dist = np.array(prev_dist)
alpha = 0.3
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

def log_prob(observed_distribution, expected_distribution):
    return np.sum(observed_distribution * np.log(1e-18 + expected_distribution)) - np.sum(observed_distribution * np.log(1e-18 + observed_distribution / sum(observed_distribution)))

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.plot(support, prev_dist, label=f'Iteration 0', marker='o')
plt.xlabel('Support')
plt.ylabel('Distribution')
for it in tqdm(range(1,4)):
    next_dist = find_next_dist(support, prev_dist, alpha)
    prev_dist = next_dist
    
    plt.plot(support, prev_dist, label=f'Iteration {it}', marker='o')
    plt.xlabel('Support')
    plt.ylabel('Distribution')

plt.plot(support, aggregate_distribution, label='Aggregate Distribution', marker='o')

plt.legend()
plt.show()