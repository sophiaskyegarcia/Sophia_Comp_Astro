#HW 6 space junk
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import solve_ivp


def equations(t, variables, M, L):
    """
    Calculates the four first-order equations describing the ball bearing's motion.
    """
    x, y, vx, vy = variables

    # Calculate the distance from the center of the rod
    r_squared = x**2 + y**2
    r = np.sqrt(r_squared)

    # Gravitational acceleration in the x and y directions
    ax = -M * x / (r_squared * np.sqrt(r_squared + (L**2 / 4)))
    ay = -M * y / (r_squared * np.sqrt(r_squared + (L**2 / 4)))

    # Return the first derivatives of x, y, vx, and vy
    return [vx, vy, ax, ay]


def calculate_orbit(M, L, t_start, t_end, num_points):
    """
    Solves the equations of motion and returns the ball bearing's position over time
    """
    # Initial position and velocity: (x, y) = (1, 0), velocity = (0, 1)
    initial_conditions = [1.0, 0.0, 0.0, 1.0]

    # Create the time values at which the solution will be calculated
    times = np.linspace(t_start, t_end, num_points)

    # Numerically solve the differential equations
    solution = solve_ivp(
        equations,
        [t_start, t_end],
        initial_conditions,
        args=(M, L),
        t_eval=times
    )

    return solution


def plot_orbit(solution):
    """
    Plots the orbit of the ball bearing with y on the vertical axis and x on the horizontal axis.
    """
    x = solution.y[0]
    y = solution.y[1]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Ball bearing's orbit")

    # Mark the starting position
    plt.plot(x[0], y[0], 'ro', label="Starting position")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Orbit of a Ball Bearing Around a Heavy Rod")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    """
    Reads command-line arguments, calculates the orbit, and displays the plot.
    """
    parser = argparse.ArgumentParser(
        description="Calculate and plot the orbit of a ball bearing around a heavy rod."
    )

    parser.add_argument("--M", type=float, default=10,
                        help="Mass of the rod (default: 10)")
    parser.add_argument("--L", type=float, default=2,
                        help="Length of the rod (default: 2)")
    parser.add_argument("--t_start", type=float, default=0,
                        help="Starting time (default: 0)")
    parser.add_argument("--t_end", type=float, default=10,
                        help="Ending time (default: 10)")
    parser.add_argument("--num_points", type=int, default=1000,
                        help="Number of points to calculate (default: 1000)")

    args = parser.parse_args()

    # Calculate the orbit using the given parameters
    solution = calculate_orbit(
        args.M,
        args.L,
        args.t_start,
        args.t_end,
        args.num_points
    )

    # Check that the numerical solver finished successfully
    if not solution.success:
        print("The calculation was unsuccessful:", solution.message)
        return

    print("Orbit calculation completed successfully.")

    # Display the orbit
    plot_orbit(solution)


if __name__ == "__main__":
    main()
