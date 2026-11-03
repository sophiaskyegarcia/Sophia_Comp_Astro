import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Constants in cgs units
SIGMA_T = 6.652e-25       # Thomson cross-section, cm^2
C = 2.998e10              # Speed of light, cm/s

R_SUN = 696340.0 * 1e5    # Solar radius, cm
R_MAX = 0.9 * R_SUN

N_ELECTRON = 1e20         # Part 1 electron density, cm^-3
SLAB_WIDTH = 1e5          # 1 km, cm
STEP_SIZE = 5000.0        # 50 m, cm

N_E_10 = 2.5e26           # Central solar electron density, cm^-3


def mean_free_path(n_e):
    """Return the mean free path in cm."""
    return 1.0 / (n_e * SIGMA_T)


def solar_density(r):
    """Return the solar electron density at radius r."""
    return N_E_10 * np.exp(-r / (0.096 * R_SUN))


def new_direction():
    """Choose a new isotropic direction in 3D."""
    cos_theta = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    sin_theta = np.sqrt(1.0 - cos_theta**2)

    ux = cos_theta
    uy = sin_theta * np.cos(phi)
    uz = sin_theta * np.sin(phi)

    return ux, uy, uz


def simulate_photon(width, n_e, step_size, save_path=False,
                    max_steps=10000000):
    """
    Follow one photon through a constant-density slab.

    The photon starts at x=0 and initially travels in the +x direction.
    At each step, the scattering probability is sigma_T * n_e * distance.
    After a scattering, a new isotropic 3D direction is selected.

    The y coordinate is saved for the animation. The z coordinate is
    also tracked, even though the animation shows only the xy projection!
    """
    x, y, z = 0.0, 0.0, 0.0

    # Initially, the photon travels straight through the slab
    ux, uy, uz = 1.0, 0.0, 0.0

    total_distance = 0.0
    number_scatterings = 0
    escaped_forward = False

    xs = [x]
    ys = [y]

    for step in range(max_steps):

        # Find distance to slab boundary in current direction.
        if ux > 0.0:
            distance_to_edge = (width - x) / ux
        elif ux < 0.0:
            distance_to_edge = -x / ux
        else:
            distance_to_edge = np.inf

        distance = min(step_size, distance_to_edge)

        # Probability of scattering during this step.
        probability = SIGMA_T * n_e * distance
        probability = min(probability, 1.0)

        if np.random.random() < probability:

            # Place the scattering somewhere within this step.
            scatter_distance = np.random.uniform(0.0, distance)

            x += ux * scatter_distance
            y += uy * scatter_distance
            z += uz * scatter_distance
            total_distance += scatter_distance

            number_scatterings += 1
            ux, uy, uz = new_direction()

            xs.append(x)
            ys.append(y)

        else:
            # The photon completes the step without scattering
            x += ux * distance
            y += uy * distance
            z += uz * distance
            total_distance += distance

            xs.append(x)
            ys.append(y)

            # Check whether the photon reached either slab boundary, end gif
            if distance_to_edge <= step_size:
                escaped_forward = (x >= width - 1e-10 * width)
                break

    else:
        raise RuntimeError(
            "Photon exceeded the maximum number of steps. "
            "Try changing the simulation parameters."
        )

    return {
        "escaped_forward": escaped_forward,
        "distance": total_distance,
        "time": total_distance / C,
        "scatterings": number_scatterings,
        "x": xs if save_path else None,
        "y": ys if save_path else None
    }


def animate_photon(result):
    """Animate the xy projection of one photon path"""
    x = np.array(result["x"]) / 1000.0
    y = np.array(result["y"]) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xlim(
        -0.05 * SLAB_WIDTH / 1000,
        1.05 * SLAB_WIDTH / 1000
    )

    y_limit = max(np.max(np.abs(y)) * 1.2, 0.1)
    ax.set_ylim(-y_limit, y_limit)

    ax.set_xlabel("Position through slab (km)")
    ax.set_ylabel("Transverse position (km)")
    ax.set_title("Photon Path Through the Slab")

    # These lines mark the two boundaries of the slab.
    ax.axvline(0, color="black", linewidth=2)
    ax.axvline(SLAB_WIDTH / 1000, color="black", linewidth=2)
    ax.grid(alpha=0.3)

    line, = ax.plot([], [], linewidth=1.5)
    point, = ax.plot([], [], "ro")

    def initialize():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(x[:frame + 1], y[:frame + 1])
        point.set_data([x[frame]], [y[frame]])
        return line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(x),
        init_func=initialize,
        interval=200,
        blit=True
    )

    ani.save("photon_trajectory.gif", writer="pillow", fps=5)

    print("Saved photon animation: photon_trajectory.gif")

    return fig, ani


def part1(number_photons, step_size=STEP_SIZE, make_animation=True):
    """Run the constant-density slab simulation."""
    print("\n========== PART 1 ==========")

    start = time.perf_counter()

    lam = mean_free_path(N_ELECTRON)
    probability = SIGMA_T * N_ELECTRON * step_size

    print(f"Electron density: {N_ELECTRON:.3e} cm^-3")
    print(f"Mean free path: {lam:.3e} cm ({lam / 100:.2f} m)")
    print(f"Slab width: {SLAB_WIDTH / 100:.2f} m")
    print(f"Step size: {step_size:.1f} cm ({step_size / 100:.1f} m)")
    print(f"Scattering probability per full step: {probability:.4f}")

    forward = 0
    backward = 0
    times = []
    scatterings = []

    for i in range(number_photons):
        result = simulate_photon(
            SLAB_WIDTH,
            N_ELECTRON,
            step_size
        )

        times.append(result["time"])
        scatterings.append(result["scatterings"])

        if result["escaped_forward"]:
            forward += 1
        else:
            backward += 1

    elapsed = time.perf_counter() - start

    print("\nSimulation results:")
    print(f"Total photons: {number_photons}")
    print(f"Forward escapes: {forward}")
    print(f"Backward escapes: {backward}")
    print(f"Transmission fraction: {forward / number_photons:.4f}")
    print(f"Reflection fraction: {backward / number_photons:.4f}")
    print(f"Average scatterings: {np.mean(scatterings):.2f}")
    print(f"Average travel time: {np.mean(times):.4e} s")
    print(f"Simulation runtime: {elapsed:.4f} seconds")

    # This histogram shows how many scatterings the the photons experienced
    # before escaping through either side of the slab.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scatterings, bins=40, edgecolor="black")
    ax.set_xlabel("Number of scatterings")
    ax.set_ylabel("Number of photons")
    ax.set_title("Photon Scattering Distribution")
    fig.tight_layout()
    fig.savefig("part1_scatterings.png", dpi=200)

    animation_figures = []
    animation_objects = []

    if make_animation:
        trajectory = simulate_photon(
            SLAB_WIDTH,
            N_ELECTRON,
            step_size,
            save_path=True
        )

        animation_fig, ani = animate_photon(trajectory)
        animation_figures.append(animation_fig)
        animation_objects.append(ani)

    return times, fig, animation_figures, animation_objects


def calibrate_travel_times(number_samples, optical_depth=10,
                           step_fraction=0.1):
    """
    Find the distribution of travel times through a slab 10 mean
    free paths thick.

    The times are stored in units of the mean free travel time,
    lambda/c. Only photons that cross the slab are kept.
    """
    print("\n========== PART 2: CALIBRATION ==========")

    start = time.perf_counter()

    # Use lambda = 1 so distances are measured in mean free paths.
    lam = 1.0
    n_e = 1.0 / (SIGMA_T * lam)
    width = optical_depth * lam
    step_size = step_fraction * lam

    travel_times = []
    total_attempts = 0

    while len(travel_times) < number_samples:
        result = simulate_photon(width, n_e, step_size)
        total_attempts += 1

        if result["escaped_forward"]:
            # Distance/lambda is time in units of lambda/c.
            travel_times.append(result["distance"] / lam)

    travel_times = np.array(travel_times)
    elapsed = time.perf_counter() - start

    print(f"Optical depth: {optical_depth}")
    print(f"Step size: {step_fraction:.3f} mean free paths")
    print(f"Successful transmitted photons: {number_samples}")
    print(f"Total attempts: {total_attempts}")
    print(f"Transmission fraction: {number_samples / total_attempts:.4f}")
    print(f"Mean dimensionless travel time: {np.mean(travel_times):.4f}")
    print(f"Calibration runtime: {elapsed:.4f} seconds")

    # This histogram gives the distribution of travel times for photons
    # that successfully cross a slab 10 mean free paths thick.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(travel_times, bins=40, density=True, edgecolor="black")
    ax.set_xlabel(r"Travel time in units of $\lambda/c$")
    ax.set_ylabel("Probability density")
    ax.set_title("Travel Times Through 10 Mean Free Paths")
    fig.tight_layout()
    fig.savefig("part2_calibration.png", dpi=200)

    return travel_times, fig


def simulate_sun(calibration_times, number_photons, number_shells=200):
    """
    Estimate photon escape times through the Sun using radial shells oooo.

    Each shell uses the local electron density to calculate its mean
    free path. A calibration time is drawn and scaled using a diffusion
    approximation. This treats the shells as successive outward steps.
    """
    print("\n========== SOLAR SIMULATION ==========")

    start = time.perf_counter()

    radial_edges = np.linspace(0.0, R_MAX, number_shells + 1)
    radial_midpoints = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    dr = R_MAX / number_shells

    densities = solar_density(radial_midpoints)
    mean_paths = mean_free_path(densities)
    optical_depths = dr / mean_paths

    print(f"Number of radial shells: {number_shells}")
    print(f"Central electron density: {solar_density(0):.3e} cm^-3")
    print(f"Density at outer boundary: {solar_density(R_MAX):.3e} cm^-3")
    print(f"Central mean free path: "
          f"{mean_free_path(solar_density(0)):.3e} cm")
    print(f"Outer mean free path: "
          f"{mean_free_path(solar_density(R_MAX)):.3e} cm")

    escape_times = np.zeros(number_photons)

    for photon in range(number_photons):
        total_time = 0.0

        for j in range(number_shells):
            # Draw a travel time from the calibration distribution.
            t10 = np.random.choice(calibration_times)

            # Scale the calibration time to the optical depth of this shell
            diffusion_time = (
                t10
                * (optical_depths[j] / 10.0)**2
                * mean_paths[j] / C
            )

            # A shell cannot be crossed faster than the light travels across it.
            minimum_time = dr / C

            shell_time = max(diffusion_time, minimum_time)
            total_time += shell_time

        escape_times[photon] = total_time

    elapsed = time.perf_counter() - start

    print("\nSolar photon results:")
    print(f"Number of simulated photons: {number_photons}")
    print(f"Mean escape time: {np.mean(escape_times):.4e} s")
    print(f"Median escape time: {np.median(escape_times):.4e} s")
    print(f"Standard deviation: {np.std(escape_times):.4e} s")

    seconds_per_year = 365.25 * 24 * 3600
    print(f"Mean escape time in years: "
          f"{np.mean(escape_times) / seconds_per_year:.4e}")
    print(f"Solar simulation runtime: {elapsed:.4f} seconds")

    # This plot shows how the electron density decreases w radius.
    # A linear y-axis is used, would a log scale have been better ?
    density_fig, density_ax = plt.subplots(figsize=(8, 5))
    density_ax.plot(radial_midpoints / R_SUN, densities)
    density_ax.set_xlabel(r"Radius ($r/R_{\odot}$)")
    density_ax.set_ylabel(r"Electron density (cm$^{-3}$)")
    density_ax.set_title("Solar Electron Density")
    density_fig.tight_layout()
    density_fig.savefig("solar_density.png", dpi=200)

    # This histogram shows the estimated escape-time distribution
    # for the simulated photons traveling through the solar shells.
    escape_fig, escape_ax = plt.subplots(figsize=(8, 5))
    escape_ax.hist(escape_times, bins=40, edgecolor="black")
    escape_ax.set_xlabel("Photon escape time (seconds)")
    escape_ax.set_ylabel("Number of photons")
    escape_ax.set_title("Solar Photon Escape Times")
    escape_fig.tight_layout()
    escape_fig.savefig("solar_escape_times.png", dpi=200)

    return escape_times, density_fig, escape_fig


def main():
    parser = argparse.ArgumentParser(
        description="HW5: Monte Carlo Radiative Transfer"
    )

    parser.add_argument(
        "--part",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Run Part 1, Part 2, or both (0)"
    )

    parser.add_argument(
        "--photons",
        type=int,
        default=1000,
        help="Number of photons to simulate"
    )

    parser.add_argument(
        "--calibration",
        type=int,
        default=1000,
        help="Number of successful calibration photons"
    )

    parser.add_argument(
        "--shells",
        type=int,
        default=200,
        help="Number of radial shells in the solar model"
    )

    parser.add_argument(
        "--step",
        type=float,
        default=STEP_SIZE,
        help="Part 1 step size in cm (default: 5000 cm)"
    )

    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Skip saving and displaying the photon animation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a random seed for reproducible results"
    )

    args = parser.parse_args()

    if args.photons < 1 or args.calibration < 1 or args.shells < 1:
        parser.error("Photon counts and number of shells must be positive.")

    if args.step <= 0:
        parser.error("Step size must be positive.")

    if args.seed is not None:
        np.random.seed(args.seed)

    overall_start = time.perf_counter()

    figures = []
    animation_objects = []

    print("\nStarting HW5: Monte Carlo Radiative Transfer")
    print("--------------------------------------------")

    if args.part in [0, 1]:
        _, part1_fig, animation_figures, part1_animations = part1(
            number_photons=args.photons,
            step_size=args.step,
            make_animation=not args.no_animate
        )

        figures.append(part1_fig)
        figures.extend(animation_figures)
        animation_objects.extend(part1_animations)

    if args.part in [0, 2]:
        calibration_times, calibration_fig = calibrate_travel_times(
            number_samples=args.calibration,
            optical_depth=10
        )

        figures.append(calibration_fig)

        _, density_fig, escape_fig = simulate_sun(
            calibration_times=calibration_times,
            number_photons=args.photons,
            number_shells=args.shells
        )

        figures.extend([density_fig, escape_fig])

    elapsed = time.perf_counter() - overall_start

    print("\n============================================")
    print("               FINAL RUN SUMMARY")
    print("============================================")
    print(f"Parts completed: {args.part if args.part != 0 else '1 and 2'}")
    print(f"Photons simulated: {args.photons}")

    if args.part in [0, 2]:
        print(f"Successful calibration samples: {args.calibration}")
        print(f"Solar radial shells: {args.shells}")

    print(f"Total runtime: {elapsed:.4f} seconds")
    print("--------------------------------------------")
    print("Output files:")

    if args.part in [0, 1]:
        print("  part1_scatterings.png")

    if args.part in [0, 2]:
        print("  part2_calibration.png")
        print("  solar_density.png")
        print("  solar_escape_times.png")

    if args.part in [0, 1] and not args.no_animate:
        print("  photon_trajectory.gif")

    print("--------------------------------------------")
    print("All simulations finished.")
    print("Displaying all plots now.")
    print("============================================\n")

    # Keep the animation objects in memory while all figures are display.
    plt.show()


if __name__ == "__main__":
    main()