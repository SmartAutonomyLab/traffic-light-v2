import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = 3        # Change this value as desired
N = 15       # Maximum value of n

# Compute partial sums
n_values = np.arange(1, N + 1)
partial_sums = []

for n in n_values:
    total = sum(x**i for i in range(1, n + 1))
    partial_sums.append(total)

log = np.log10(partial_sums)

# Plot
plt.figure()
plt.plot(n_values, log )
plt.xlabel("n")
plt.ylabel("Sum_{i=1}^n x^i")
plt.title("Partial Sum of Geometric Series")
plt.show()
