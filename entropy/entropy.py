# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn (Original code)
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate entropy
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities + (probabilities == 0)))

# Part 1: x = number of options, y = entropy
x1 = np.arange(1, 51)
y1 = np.array([calculate_entropy(np.ones(i) / i) for i in x1])


# Part 2: Redefine uniformity level (1 is completely uniform, 0 is completely non-uniform)
uniformity_levels = np.linspace(0, 1, 20)
x2 = uniformity_levels
y2 = []
for level in uniformity_levels:
    # Create a distribution that becomes more uniform as level increases
    non_uniform_part = (1 - level) / 49  # Non-uniform distribution part
    probabilities = np.array([non_uniform_part] * 49 + [level + non_uniform_part])
    y2.append(calculate_entropy(probabilities))
y2 = np.array(y2)

# Part 3: Redefine for different option counts
option_counts = [2, 3, 4, 5, 6, 7, 8]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
x3 = uniformity_levels
y3_dict = {}
for count in option_counts:
    y3 = []
    for level in uniformity_levels:
        non_uniform_part = (1 - level) / (count - 1)  # Non-uniform distribution part
        probabilities = np.array([non_uniform_part] * (count - 1) + [level + non_uniform_part])
        y3.append(calculate_entropy(probabilities))
    y3_dict[count] = y3

# Plotting the updated graphs
plt.figure(figsize=(18, 6))

# Plot 1 remains the same
plt.subplot(1, 3, 1)
plt.plot(x1, y1, label='Options vs Entropy')
plt.xlabel('Number of Options')
plt.ylabel('Entropy')
plt.title('Options vs Entropy')

# Plot 2: Updated
plt.subplot(1, 3, 2)
plt.plot(x2, y2, label='Uniformity vs Entropy')
plt.xlabel('Uniformity Level (1 is perfectly uniform)')
plt.ylabel('Entropy')
plt.title('Uniformity vs Entropy')

# Plot 3: Updated
plt.subplot(1, 3, 3)
for count in option_counts:
    plt.plot(x3, y3_dict[count], color=colors[option_counts.index(count)], label=f'Option Count {count}')
plt.xlabel('Uniformity Level (1 is perfectly uniform)')
plt.ylabel('Entropy')
plt.title('Aggregated Variable vs Entropy')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("e.png")
plt.savefig("e.pdf")
