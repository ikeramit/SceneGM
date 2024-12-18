
import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 6.626e-34  # Planck's constant (J·s)
c = 3e8        # Speed of light (m/s)
k_B = 1.38e-23 # Boltzmann constant (J/K)

def planck_law(wavelength, temperature):
    """
    Computes spectral radiance using Planck's Law.

    Parameters:
        wavelength (ndarray): Wavelengths in meters.
        temperature (float): Temperature in Kelvin.

    Returns:
        ndarray: Spectral radiance (W/m^2/sr/m).
    """
    term1 = (2 * h * c**2) / (wavelength**5)
    term2 = 1 / (np.exp((h * c) / (wavelength * k_B * temperature)) - 1)
    return term1 * term2

def transmissivity(distance, absorption_coefficient=0.01):
    """
    Computes atmospheric transmissivity based on distance.

    Parameters:
        distance (float): Distance from the object to the camera (meters).
        absorption_coefficient (float): Absorption coefficient (depends on atmosphere).

    Returns:
        float: Transmissivity factor (0 to 1).
    """
    return np.exp(-absorption_coefficient * distance)

def total_radiance(temperature, distance, emissivity=1.0, absorption_coefficient=0.01, background_temp=300):
    """
    Computes total radiance measured by the camera.

    Parameters:
        temperature (float): Object temperature (K).
        distance (float): Distance to the object (m).
        emissivity (float): Emissivity of the object (0 to 1).
        absorption_coefficient (float): Atmospheric absorption coefficient.
        background_temp (float): Background temperature (K).

    Returns:
        float: Total radiance measured by the camera (W/m^2/sr).
    """
    # Wavelength range for thermal cameras (8 µm to 14 µm)
    wavelengths = np.linspace(8e-6, 14e-6, 1000)  # in meters

    # Radiance from the object
    radiance_object = np.trapz(planck_law(wavelengths, temperature), wavelengths)

    # Radiance from the background
    radiance_background = np.trapz(planck_law(wavelengths, background_temp), wavelengths)

    # Atmospheric transmissivity
    tau = transmissivity(distance, absorption_coefficient)

    # Total radiance measured by the camera
    radiance_measured = tau * (emissivity * radiance_object / distance**2) + (1 - tau) * radiance_background
    return radiance_measured

# Example: Compute radiance for a point at a distance
object_temp = 350       # Temperature of the object (K)
distance = 50           # Distance to the object (m)
emissivity = 0.95       # Emissivity of the object
absorption_coefficient = 0.01  # Atmospheric absorption coefficient
background_temp = 290   # Background temperature (K)

# Compute radiance
radiance = total_radiance(object_temp, distance, emissivity, absorption_coefficient, background_temp)
print(f"Measured Radiance at {distance} m: {radiance:.2e} W/m^2/sr")

# Plot radiance vs. distance
distances = np.linspace(1, 100, 100)  # Range of distances (m)
radiances = [total_radiance(object_temp, d, emissivity, absorption_coefficient, background_temp) for d in distances]

plt.figure(figsize=(8, 6))
plt.plot(distances, radiances, label='Measured Radiance', color='red')
plt.xlabel('Distance (m)')
plt.ylabel('Radiance (W/m²/sr)')
plt.title('Radiance vs. Distance')
plt.grid(True)
plt.legend()
plt.show()


# Plot radiance vs. temperature
temperatures = np.linspace(250, 400, 100)  # Range of temperatures (K)
radiances = [total_radiance(temp, distance, emissivity, absorption_coefficient, background_temp) for temp in temperatures]

plt.figure(figsize=(8, 6))
plt.plot(temperatures, radiances, label='Measured Radiance', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Radiance (W/m²/sr)')
plt.title('Radiance vs. Temperature')
plt.grid(True)
plt.legend()
plt.show()
