import os
import json

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

import matplotlib.pyplot as plt

def plot_frequencies(frequencies):
    ys = [sum(freqs.values()) for freqs in frequencies]
    xs = range(len(frequencies))

    plt.plot(xs, ys)
    plt.xlabel("Day")
    plt.ylabel("Total Players")
    plt.title(f"Total Players Over Time, average ~ {sum(ys)/len(ys):.0f}")
    plt.show()


if __name__ == "__main__":
    all_frequencies = iterate_over_days()
    plot_frequencies(all_frequencies)