import math


def E(x, n=1000):
    """
    Calculate E(x) = integral from 0 to x of exp(-t^2) dt
    using the trapezoidal rule.
    """

    h = x / n

    # Trapezoidal rule
    total = 0.5 * (math.exp(0) + math.exp(-x**2))

    for i in range(1, n):
        t = i * h
        total += math.exp(-t**2)

    return h * total


# Part (a)
# Calculate E(x) for x = 0, 0.1, 0.2, ..., 3.0

x_values = []
E_values = []

print("   x          E(x)")
print("----------------------")

for i in range(31):
    x = i * 0.1
    value = E(x)

    x_values.append(x)
    E_values.append(value)

    print(f"{x:5.1f}     {value:.8f}")


# Part (b)
# Make a simple graph using characters in the terminal

print("\nGraph of E(x)")
print("-------------")

max_value = max(E_values)
graph_width = 60

for x, value in zip(x_values, E_values):
    # Scale the value to fit the graph
    number_of_stars = int((value / max_value) * graph_width)

    print(f"{x:3.1f} | {'*' * number_of_stars}")
