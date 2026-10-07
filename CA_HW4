import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


#I used this code on a few different fits files from the website 
# and you can change it to whichever you have downloaded below. This 
# one is just the very first one on the website 
#This code automatically prints out the plots one after the other

# Change this to the name/path of your FITS file
filename = "tic0000091961.fits"

# Number of Fourier coefficients to keep in the reconstruction.
# to try: 5, 10, 20, 50, 100.
N_COEFFICIENTS = 20

# How large a gap (in units of the normal time step) counts
# as a break between observing epochs.
GAP_FACTOR = 5



# READ THE TESS DATA


hdul = fits.open(filename)

times = hdul[1].data["times"]
fluxes = hdul[1].data["fluxes"]
ferrs = hdul[1].data["ferrs"]

hdul.close()

# Convert to NumPy arrays
times = np.asarray(times, dtype=float)
fluxes = np.asarray(fluxes, dtype=float)
ferrs = np.asarray(ferrs, dtype=float)



# REMOVE BAD DATA


good = np.isfinite(times) & np.isfinite(fluxes)

times = times[good]
fluxes = fluxes[good]
ferrs = ferrs[good]

# Make sure the data are sorted by time
order = np.argsort(times)

times = times[order]
fluxes = fluxes[order]
ferrs = ferrs[order]



# PLOT THE FULL LIGHT CURVE


plt.figure(figsize=(12, 5))

plt.plot(times, fluxes, ".", markersize=2)

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Full TESS Light Curve")

plt.tight_layout()
plt.show()



# FIND OBSERVING EPOCHS


# Calc the time between consecutive observations
dt = np.diff(times)

# typical time step
typical_dt = np.median(dt)

print("Typical time step:", typical_dt)

# A large gap indicates that the telescope stopped observing
# and later returned to the star.
gap_indices = np.where(dt > GAP_FACTOR * typical_dt)[0]

print("Number of large gaps:", len(gap_indices))

# Start and end indices of each observing epoch
epoch_starts = np.r_[0, gap_indices + 1]
epoch_ends = np.r_[gap_indices, len(times) - 1]

print("\nObserving epochs:")

for i, (start, end) in enumerate(zip(epoch_starts, epoch_ends)):
    number_of_points = end - start + 1
    duration = times[end] - times[start]

    print(
        f"Epoch {i + 1}: "
        f"{number_of_points} points, "
        f"duration = {duration:.3f}"
    )


# CHOOSE THE EPOCH WITH THE MOST DATA


epoch_sizes = epoch_ends - epoch_starts + 1

best_epoch = np.argmax(epoch_sizes)

start = epoch_starts[best_epoch]
end = epoch_ends[best_epoch]

epoch_times = times[start:end + 1]
epoch_fluxes = fluxes[start:end + 1]

print("\nUsing epoch:", best_epoch + 1)
print("Number of points:", len(epoch_times))
print("Start time:", epoch_times[0])
print("End time:", epoch_times[-1])



# PLOT THE SELECTED EPOCH


plt.figure(figsize=(12, 5))

plt.plot(epoch_times, epoch_fluxes, ".", markersize=3)

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Selected TESS Observing Epoch")

plt.tight_layout()
plt.show()



# NORMALIZE THE LIGHT CURVE


# Remove the average flux so that the zero-frequency component
# doesnt dominate the Fourier spectrum.

mean_flux = np.mean(epoch_fluxes)

normalized_flux = epoch_fluxes / mean_flux

# Subtract the average value
signal = normalized_flux - np.mean(normalized_flux)



# CHECK THE SAMPLING


dt_epoch = np.diff(epoch_times)

median_dt = np.median(dt_epoch)

print("\nSampling information:")
print("Median time step:", median_dt)
print("Minimum time step:", np.min(dt_epoch))
print("Maximum time step:", np.max(dt_epoch))

# Look for missing time steps
missing = np.where(dt_epoch > 1.5 * median_dt)[0]

print("Number of possible missing time steps:", len(missing))

if len(missing) > 0:
    print("Possible missing observations at:")

    for index in missing[:20]:
        print(
            f"  Between {epoch_times[index]:.6f} "
            f"and {epoch_times[index + 1]:.6f}"
        )



# FOURIER TRANSFORM


N = len(signal)

# Fourier transform
fourier = np.fft.fft(signal)

# Frequencies corresponding to the Fourier coefficients
frequencies = np.fft.fftfreq(N, d=median_dt)

# Power spectrum
power = np.abs(fourier) ** 2



# ONLY PLOT POSITIVE FREQUENCIES


positive = frequencies > 0

positive_frequencies = frequencies[positive]
positive_power = power[positive]

plt.figure(figsize=(12, 5))

plt.plot(
    positive_frequencies,
    positive_power,
    ".",
    markersize=3
)

plt.xlabel("Frequency (1/time)")
plt.ylabel("Power")
plt.title("Fourier Power Spectrum")

plt.tight_layout()
plt.show()



# FIND THE STRONGEST FOURIER COEFFICIENTS


# Ignore the zero-frequency coefficient
coefficient_indices = np.arange(N)

nonzero_indices = coefficient_indices[1:]

# Sort Fourier coefficients by their strength
sorted_indices = nonzero_indices[
    np.argsort(np.abs(fourier[nonzero_indices]))[::-1]
]

# Keep only the strongest coefficients
keep = sorted_indices[:N_COEFFICIENTS]

# Always keep the zero-frequency coefficient as well.
# It represents the average level of the signal.
keep = np.r_[0, keep]

print("\nStrongest Fourier coefficients:")

for index in keep:
    print(
        f"Index = {index:6d}, "
        f"frequency = {frequencies[index]:12.6f}, "
        f"amplitude = {np.abs(fourier[index]):12.6f}"
    )



# RECONSTRUCT LIGHT CURVE


# Make an array containing only the Fourier coefficients
# that we decided to keep.

filtered_fourier = np.zeros_like(fourier)

filtered_fourier[keep] = fourier[keep]

# Inverse Fourier transform
reconstructed_signal = np.fft.ifft(filtered_fourier).real

# Add the mean back
reconstructed_flux = (
    reconstructed_signal + np.mean(normalized_flux)
)



# PLOT ORIGINAL VS RECONSTRUCTED LIGHT CURVE


plt.figure(figsize=(12, 5))

plt.plot(
    epoch_times,
    normalized_flux,
    ".",
    markersize=2,
    alpha=0.5,
    label="Original"
)

plt.plot(
    epoch_times,
    reconstructed_flux,
    "r-",
    linewidth=2,
    label=f"Using {N_COEFFICIENTS} Fourier coefficients"
)

plt.xlabel("Time")
plt.ylabel("Normalized Flux")

plt.title("Fourier Reconstruction of the Eclipsing Binary")

plt.legend()

plt.tight_layout()
plt.show()



# PLOT ONLY A SMALL SECTION


# Looking at a smaller region makes it easier to see whether
# the Fourier reconstruction actually captures the eclipse.

number_to_show = min(1000, len(epoch_times))

plt.figure(figsize=(12, 5))

plt.plot(
    epoch_times[:number_to_show],
    normalized_flux[:number_to_show],
    ".",
    markersize=3,
    alpha=0.6,
    label="Original"
)

plt.plot(
    epoch_times[:number_to_show],
    reconstructed_flux[:number_to_show],
    "r-",
    linewidth=2,
    label="Fourier reconstruction"
)

plt.xlabel("Time")
plt.ylabel("Normalized Flux")

plt.title("Close-up of Fourier Reconstruction")

plt.legend()

plt.tight_layout()
plt.show()



# LINEAR INTERPOLATION OF MISSING DATA


# Create a completely evenly spaced time array.

regular_times = np.arange(
    epoch_times[0],
    epoch_times[-1],
    median_dt
)

# Interpolate the light curve onto the regular time grid.
regular_flux = np.interp(
    regular_times,
    epoch_times,
    normalized_flux
)

# Remove the average
regular_signal = regular_flux - np.mean(regular_flux)



# FOURIER TRANSFORM POST INTERPOLATION


N_regular = len(regular_signal)

fourier_regular = np.fft.fft(regular_signal)

frequencies_regular = np.fft.fftfreq(
    N_regular,
    d=median_dt
)

power_regular = np.abs(fourier_regular) ** 2



# COMPARE THE TWO POWER SPECTRA


positive_regular = frequencies_regular > 0

plt.figure(figsize=(12, 5))

plt.plot(
    positive_frequencies,
    positive_power,
    label="Original sampling",
    alpha=0.7
)

plt.plot(
    frequencies_regular[positive_regular],
    power_regular[positive_regular],
    label="After interpolation",
    alpha=0.7
)

plt.xlabel("Frequency (1/time)")
plt.ylabel("Power")

plt.title("Fourier Spectrum Before and After Interpolation")

plt.legend()

plt.tight_layout()
plt.show()



# PRINT THE DOMINANT FREQUENCY


# Find the strongest frequency in the original data

strongest = np.argmax(positive_power)

dominant_frequency = positive_frequencies[strongest]

period = 1 / dominant_frequency

print("\nDominant frequency:", dominant_frequency)
print("Estimated period:", period)




print("\nFourier analysis complete.")
