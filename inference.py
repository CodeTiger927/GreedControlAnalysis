import numpy as np
calibrate = -22693.755455660186

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
# data = [dist / sum(dist) for dist in data]
optimal = np.array(list(i + 1 for i in range(len(aggregate_distribution)))) / sum(i + 1 for i in range(len(aggregate_distribution)))

human_data = [102,77,83,89,87,69,150,84,111,51,87,71,65,59,67,74,64,70,53,79,50,91,58,56,87,52,62,50,53,57,63,61,77,47,58,85,69,45,52,59,55,60,65,86,70,61,50,64,81,116]
human_data = np.array(human_data, dtype=np.float64)
human_data = support * human_data
human_distribution = human_data / sum(human_data)


print(support, aggregate_distribution, optimal)

print("GARBAGE:", sum(log_prob(obs, np.array([1/len(support) for _ in obs])) for obs in data[1:]))
print("OPTIMAL:", sum(log_prob(obs, optimal) for obs in data[1:]))
print("CHEATING:", sum(log_prob(obs, aggregate_distribution) for obs in data[1:]))
print("PERFECT:", sum(log_prob(obs, obs / sum(obs)) for obs in data[1:]))

print("Loaded data")

def find_next_dist(support, prev_dist, alpha): 
    next_dist = np.ones_like(prev_dist) / len(prev_dist)
    for _ in range(10):
        next_dist = next_dist * np.array(support) / (alpha * next_dist + (1 - alpha) * prev_dist)
        next_dist /= np.sum(next_dist)

    return next_dist

def get_log_prob(observed, initial_distribution, alpha, beta, depth, support, store_distributions=False):
    """
    Calculate the log probability of the observed distributions given the initial distribution
    and the alpha value

    Parameters:
    observed (list): The observed distributions
    initial_distribution (list): The initial distribution of the players
    depth (int): The number of days to consider
    alpha (float): The alpha value for the model
    beta (float): The beta value for the model
    support (list): The support for the distributions

    Returns:
    float: The log probability of the observed distributions
    """
    import numpy as np
    distributions = [initial_distribution]
    for i in range(1, depth+1):
        next_dist = find_next_dist(support, distributions[-1], alpha)
        distributions.append(next_dist)

    predicted_distribution = np.sum([beta**i * dist for i, dist in enumerate(distributions)], axis=0)
    predicted_distribution = predicted_distribution / np.sum(predicted_distribution)

    if store_distributions:
        return log_prob(observed, predicted_distribution), predicted_distribution

    return log_prob(observed, predicted_distribution)

def calculate_log_prob(data, get_initial_distribution, get_alpha, get_beta, depth, show_pbar=False, store_distributions=False):
    """
    Calculate the log probability of the observed distributions given the initial distribution
    and the alpha value

    Parameters:
    data (list): The observed distributions
    get_initial_distribution (function): A function that returns the initial distribution given the data and current day
    get_alpha (function): A function that returns the alpha value given the data and current day
    get_beta (function): A function that returns the beta value given the data and current day
    depth (int): The number of days to consider

    Returns:
    float: The log probability of the observed distributions
    """
    import tqdm as tqdm
    log_probs = []
    aggregate = []
    t = tqdm.tqdm if show_pbar else lambda x: x
    for i in t(range(1, len(data))):
        initial_distribution = get_initial_distribution(data, i)
        alpha = get_alpha(data, i)
        beta = get_beta(data, i)
        if store_distributions:
            log_prob, _ = get_log_prob(data[i], initial_distribution, alpha, beta, depth, support, store_distributions)
            aggregate.append(_)
        else:
            log_prob = get_log_prob(data[i], initial_distribution, alpha, beta, depth, support, store_distributions)
        log_probs.append(log_prob)
    print(f"Log prob: {sum(log_probs)}")
    if store_distributions:
        return sum(log_probs), aggregate
    return sum(log_probs)

def initial_distribution_const(dist):
    return lambda data, i: dist

def initial_distribution_previous_day(data, i):
    if i == 0:
        assert False
    return data[i-1]

def mixed_initial_distribution(dist, gamma):
    return lambda data, i: gamma * dist + (1 - gamma) * data[i-1]

def initial_distribution_exp_wavg(alpha=0.9):
    def initial_distribution_wavg(data, i):
        if i == 0:
            assert False
        weights = np.array([alpha**(i-j) for j in range(i)])
        weights = weights / weights.sum()
        weighted_avg = np.sum(data[:i] * weights[:, None], axis=0)
        return weighted_avg / weighted_avg.sum()
    return initial_distribution_wavg
    
def alpha_constant(alpha):
    return lambda data, i: alpha

import matplotlib.pyplot as plt
plt.plot(support, human_distribution, label='Human Distribution', marker='o')
plt.show()

def evaluate_params(params):
    alpha, beta = params
    print(alpha)
    return calculate_log_prob(data, initial_distribution_exp_wavg(1.0), alpha_constant(alpha), alpha_constant(beta), 25, show_pbar=False)

alphas = np.linspace(0.1, 1.0, 10)
betas = np.linspace(0.1, 1.0, 10)

results = [[evaluate_params((alpha, beta)) for beta in betas] for alpha in alphas]

print(results)

import numpy as np
import matplotlib.pyplot as plt

# Create a 2D array of the results
results = np.array(results)

# Create a meshgrid for the alphas and gammas
alphas, betas = np.meshgrid(alphas, betas)

baseline = sum(log_prob(obs, optimal) for obs in data[1:])

# Plot the results
plt.figure(figsize=(10, 6))
plt.contourf(alphas, betas, results, levels=20, cmap='viridis')
plt.colorbar()
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Log Probability of Observed Distributions')

# Plot the baseline contour
plt.contour(alphas, betas, results, levels=[baseline], colors='red', linestyles='dashed')

# Find the peak
peak_idx = np.unravel_index(np.argmax(results), results.shape)
peak_alpha = alphas[peak_idx]
peak_gamma = betas[peak_idx]

# Plot the peak point
plt.plot(peak_alpha, peak_gamma, 'ro', label='Peak')
plt.legend()

print(alphas, betas, results)

# Show the plot

plt.show()

best_alpha = 0.5
best_beta = 0.4

score, agg_pred = calculate_log_prob(data, initial_distribution_exp_wavg(1.0), alpha_constant(best_alpha), alpha_constant(best_beta), 25, show_pbar=False, store_distributions=True)

agg_pred = np.array(agg_pred)
agg_pred = np.sum(agg_pred, axis=0) / len(agg_pred)

plt.plot(support, human_distribution, label='Human Distribution', marker='o')
plt.plot(support, aggregate_distribution, label='Aggregate Distribution', marker='o')
plt.plot(support, agg_pred, label='Predicted Distribution', marker='o')
plt.legend()

plt.show()
