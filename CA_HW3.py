from scipy.constants import G

# Constants
M = 5.972e24       # Mass of Earth (kg)
m = 7.348e22       # Mass of Moon (kg)
R = 3.844e8        # Earth-Moon distance (m)
omega = 2.662e-6   # Angular velocity (s^-1)


# Function we want to find the root of:
#
# GM/r^2 - Gm/(R-r)^2 = omega^2 * r
#
# Rearranged so that f(r) = 0:
def f(r):
    return G * M / r**2 - G * m / (R - r)**2 - omega**2 * r


# Derivative of f(r), needed for Newton's method
def df(r):
    return -2 * G * M / r**3 - 2 * G * m / (R - r)**3 - omega**2


# Starting guess
r = 3.26e8

# Newton's method
tolerance = 1e-6
max_iterations = 100

for i in range(max_iterations):
    r_new = r - f(r) / df(r)

    if abs(r_new - r) < tolerance:
        r = r_new
        break

    r = r_new


# Print the result
print("Distance from Earth to L1 point:")
print(f"{r:.6e} meters")
print(f"{r / 1000:.3f} kilometers")
print(f"Number of iterations: {i + 1}")
