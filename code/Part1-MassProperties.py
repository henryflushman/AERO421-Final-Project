import os
os.system('cls')
import numpy as np

from Functions.RectangularMomentofInertia import momentInertiaRecPri

bus_mass = 500 #[kg]
bus_dimensions = np.array([2.0,2.0,2.0]) #[m]
bus_distance_from_center = np.array([0,0,0]) #[m]

sensor_mass = 100 #[kg]
sensor_dimensions = np.array([0.25,0.25,1.0]) #[m]
sensor_distance_from_center = np.array([0,0,1.5]) #[m]

solar_panel_mass = 20 #[kg]
solar_panel_dimensions = np.array([2.0,3.0,0.05]) #[m]
solar_panel_A_distance_from_center = np.array([0,2.5,0]) #[m]
solar_panel_B_distance_from_center = np.array([0,-2.5,0]) #[m]

small_sat_normal = np.array([
    [bus_mass,*bus_dimensions,*bus_distance_from_center],
    [sensor_mass,*sensor_dimensions,*sensor_distance_from_center],
    [solar_panel_mass,*solar_panel_dimensions,*solar_panel_A_distance_from_center],
    [solar_panel_mass,*solar_panel_dimensions,*solar_panel_B_distance_from_center]])

I_small_sat_normal,center_of_mass = momentInertiaRecPri(small_sat_normal)

print('------Normal Mode------')

print('\nMoment of Inertia Matrix =\n', I_small_sat_normal, "[kgm^2]")
print('\nCenter of Mass =', center_of_mass, '[m]')
print('\nTotal Mass =',sum(small_sat_normal[:,0]))

small_sat_detumble = np.array([[(bus_mass + sensor_mass + 2*solar_panel_mass),*bus_dimensions,*bus_distance_from_center]])

I_small_sat_detumble,center_of_mass = momentInertiaRecPri(small_sat_detumble)

print('\n------Detumble Mode------')

print('\nMoment of Inertia Matrix =\n', I_small_sat_detumble, "[kgm^2]")
print('\nCenter of Mass =', center_of_mass, '[m]')
print('\nTotal Mass =',sum(small_sat_detumble[:,0]))