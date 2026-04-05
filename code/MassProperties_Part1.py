import numpy as np

totalMass = 640                 # kg
spacecraftOrigin = [0, 0, 0]    # m

# ——— Mass Properties During the Detumbling Phase —————————————————————————

detumbleCOM = spacecraftOrigin

detumbleMMOI = (1/12) * totalMass * (2**2 + 2**2)   # kg*m**2
detumbleMMOI_matrix = np.eye(3) * detumbleMMOI      # kg*m**2

print("Detumbling Phase:")
print("Center of Mass:", detumbleCOM, "m")
print(f"Moment of Inertia: {detumbleMMOI:.3f} kg*m\u00B2")
print(f"Inertia Matrix:\n{detumbleMMOI_matrix} kg*m\u00B2")


# ——— Mass Properties During Nominal Operations ——————————————————————————

busCOM = spacecraftOrigin   # m
busMass = 500               # kg

sensorCOM = [0, 0, 1.5]     # m
sensorMass = 20             # kg
