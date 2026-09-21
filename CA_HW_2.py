import math
import argparse


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


# Set up command-line arguments
parser = argparse.ArgumentParser(
    description="Calculate E(x) using the trapezoidal rule."
)

parser.add_argument(
    "--max-x",
    type=float,
    default=3.0,
    help="Maximum x value (default: 3.0)"
)

parser.add_argument(
    "--step",
    type=float,
    default=0.1,
    help="Step size for x values (default: 0.1)"
)

parser.add_argument(
    "--n",
    type=int,
    default=1000,
    help="Number of trapezoids (default: 1000)"
)

args = parser.parse_args()


# Calculate E(x) for the specified range

x_values = []
E_values = []

print("   x          E(x)")
print("----------------------")

x = 0.0

while x <= args.max_x + 1e-10:
    value = E(x, args.n)

    x_values.append(x)
    E_values.append(value)

    print(f"{x:5.1f}     {value:.8f}")

    x += args.step


# Make a simple graph using characters in the terminal

print("\nGraph of E(x)")
print("-------------")

max_value = max(E_values)
graph_width = 60

for x, value in zip(x_values, E_values):
    # Scale the value to fit the graph
    number_of_stars = int((value / max_value) * graph_width)

    print(f"{x:3.1f} | {'*' * number_of_stars}")

