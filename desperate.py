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

human_data = [102,77,83,89,87,69,150,84,111,51,87,71,65,59,67,74,64,70,53,79,50,91,58,56,87,52,62,50,53,57,63,61,77,47,58,85,69,45,52,59,55,60,65,86,70,61,50,64,81,116]
human_data = np.array(human_data, dtype=np.float64)
human_data = support * human_data
human_distribution = human_data / sum(human_data)


print(support, aggregate_distribution, optimal)

print("GARBAGE:", sum(log_prob(obs, np.array([1/len(support) for _ in obs])) for obs in data[1:]))
print("OPTIMAL:", sum(log_prob(obs, optimal) for obs in data[1:]))
print("GUESSING AVERAGE:", sum(log_prob(obs, aggregate_distribution) for obs in data[1:]))
print("UB:", sum(log_prob(obs, obs) for obs in data[1:]))

print("Loaded data")

def find_next_dist(support, prev_dist, alpha): 
    next_dist = np.ones_like(prev_dist) / len(prev_dist)
    for _ in range(50):
        next_dist = next_dist * np.array(support) / (alpha * next_dist + (1 - alpha) * prev_dist)
        next_dist /= np.sum(next_dist)

    return next_dist

def get_log_prob(observed, initial_distribution, alpha, beta, depth, support):
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

    predicted_distribution = np.sum([alpha**i * dist for i, dist in enumerate(distributions)], axis=0)
    predicted_distribution = predicted_distribution / np.sum(predicted_distribution)
    predicted_distribution = beta * optimal + (1 - beta) * predicted_distribution

    return log_prob(observed, predicted_distribution)

def calculate_log_prob(data, get_initial_distribution, get_alpha, get_beta, depth, show_pbar=False):
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
    t = tqdm.tqdm if show_pbar else lambda x: x
    for i in t(range(1, len(data))):
        initial_distribution = get_initial_distribution(data, i)
        alpha = get_alpha(data, i)
        beta = get_beta(data, i)
        log_prob = get_log_prob(data[i], initial_distribution, alpha, beta, depth, support)
        log_probs.append(log_prob)
    print(f"Log prob: {sum(log_probs)}")
    return sum(log_probs)

def initial_distribution_const(dist):
    return lambda data, i: dist

def initial_distribution_previous_day(data, i):
    if i == 0:
        assert False
    return data[i-1]

def initial_distribution_previous_day_plus_const(dist, gamma):
    def initial_distribution_prev_day_plus_const(data, i):
        if i == 0:
            assert False
        return gamma * (data[i-1] - dist) + dist
    return initial_distribution_prev_day_plus_const

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
    alpha = params
    print(alpha)
    return calculate_log_prob(data, initial_distribution_previous_day_plus_const(human_distribution, 0), alpha_constant(alpha), alpha_constant(0.0), 25, show_pbar=False)

alphas = np.linspace(0.0, 1.0, 20)

results = [evaluate_params(alpha) for alpha in alphas]

print(results)

import matplotlib.pyplot as plt
plt.plot(alphas, results)
plt.title("Log Probability vs Alpha")
plt.xlabel("Alpha")
plt.ylabel("Log Probability")

plt.show()
