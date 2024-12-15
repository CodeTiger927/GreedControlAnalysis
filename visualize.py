logits = [-3.82962089, -5.06451978, -5.10889174, -5.01058222, -7.05660008, -7.4856798,
          -7.22286235, -7.02582841, -8.20970861, -6.67982194, -6.28387514, -5.18756725,
          -5.00508021, -5.8233404, -4.24688764, -4.76308475, -4.17512489, -5.57374529,
          -3.91234011, -3.80773304, -3.73525749, -4.35333543, -2.93720926, -3.69016372,
          -3.44719431, -3.47258159, -3.56231031, -3.13677659, -3.24608546, -3.27814819,
          -2.93114001, -2.85291265, -3.13590007, -2.7839108, -2.98809805, -3.52736942,
          -2.28740263, -2.45362343, -3.17384084, -3.24768847, -2.97205488, -3.02492105,
          -2.83753924, -3.11744388, -2.86131524, -3.06537654, -2.46706246, -3.10075003,
          -3.00788488, -2.18185517]

logits = [-3.51462779, -4.65475004, -4.90111588, -5.54173532, -5.38772075, -5.90615362,
          -6.76553995, -4.94107191, -5.64118455, -4.79241114, -4.66841558, -4.31903622,
          -5.15065824, -4.41346794, -5.16744638, -7.69225361, -5.15198365, -5.17662284,
          -5.96932895, -4.98874372, -6.7767405, -5.43577696, -2.67226787, -5.44473193,
          -2.89508945, -5.2895927, -2.83483181, -2.72647653, -2.731788, -3.05354215,
          -5.99680522, -2.36410535, -2.64920954, -2.3855448, -2.48764788, -3.22914789,
          -2.00009213, -2.11174575, -3.83181317, -2.85449002, -2.46101581, -2.64452353,
          -2.30524058, -3.82258496, -2.35603286, -3.07628781, -2.08021099, -4.13598541,
          -2.53706668, -1.85840716]

from scipy.special import softmax

probs = softmax(logits)

support = list(range(1, 51))

import numpy as np
optimal = support / np.sum(support)
# probs -= optimal
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(support, probs)
# Add title and labels
plt.title('Predicted Naive Numbers Difference from Optimal (Total)')
plt.xlabel('Items, Chosen Numbers')
plt.ylabel('Frequencies')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
plt.xticks(support, support)

plt.show()