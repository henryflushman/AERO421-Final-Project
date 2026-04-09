# ===========================================
# AERO 421 - Final Project
#   - Part 1: Mass Properties
#
# Written by Jackson Mehiel
# Colaborators:
#   - Nick Schaeffer
#   - Henry Flushman
# ===========================================

# From Repository
import satellite.SatelliteObject as sat

# === Analysis ==============================
small_sat_normal = sat.SatelliteObject(name="MehielSat")

I_small_sat_normal = small_sat_normal.J
center_of_mass = small_sat_normal.com

print('------Normal Mode------')

print('\nMoment of Inertia Matrix =\n', I_small_sat_normal, "[kgm^2]")
print('\nCenter of Mass =', center_of_mass, '[m]')
print('\nTotal Mass =', small_sat_normal.totalMass)

small_sat_detumble = sat.SatelliteObject(name="MehielSat_Detumble")

I_small_sat_detumble = small_sat_detumble.J
center_of_mass = small_sat_detumble.com

print('\n------Detumble Mode------')

print('\nMoment of Inertia Matrix =\n', I_small_sat_detumble, "[kgm^2]")
print('\nCenter of Mass =', center_of_mass, '[m]')
print('\nTotal Mass =', small_sat_detumble.totalMass)