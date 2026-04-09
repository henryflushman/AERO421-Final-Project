import os
os.system('cls')
import numpy as np

def momentInertiaRecPri(sat_data):
    '''
    Built only for numbers of rectangular prism
    
    sat_data = [mass,x_dim,y_dim,z_dim,rx,ry,rz] by number of bodies
    
    '''

    masses = sat_data[:,0]
    #print(masses)
    dimensions = sat_data[:,1:4]
    #print(dimensions)
    distance_from_center = sat_data[:,4:7] #<--- Need this for parallel axis theorem
    #print(distance_from_center)

    n = range(len(masses))

    def centerMass(masses,distance_from_center):

        total_mass = sum(masses[i] for i in n)

        Cx = sum(masses[i] * distance_from_center[i,0] for i in n) / total_mass
        Cy = sum(masses[i] * distance_from_center[i,1] for i in n) / total_mass
        Cz = sum(masses[i] * distance_from_center[i,2] for i in n) / total_mass

        return np.array([Cx, Cy, Cz])
    
    def crossR(dimensions): #<--- distances = [x,y,z]

        return np.array([
            [0, -dimensions[2], dimensions[1]],
            [dimensions[2], 0, -dimensions[0]],
            [-dimensions[1], dimensions[0], 0]])
    
    def I_cm_rectangular_prism(masses, dimensions):

        Ixx = (1/12)*masses*(dimensions[1]**2 + dimensions[2]**2)
        Iyy = (1/12)*masses*(dimensions[0]**2 + dimensions[2]**2)
        Izz = (1/12)*masses*(dimensions[0]**2 + dimensions[1]**2)

        return np.array([
            [Ixx,0,0],
            [0,Iyy,0],
            [0,0,Izz]])
    
    I_cm_vector = [I_cm_rectangular_prism(masses[i], dimensions[i]) for i in n]

    def parallelAxis(mass, Icm, rx):

        return Icm - mass * (rx @ rx)
    
    I_vector = [parallelAxis(masses[i], I_cm_vector[i], crossR(distance_from_center[i] - centerMass(masses, distance_from_center))) for i in n]

    I_total = sum(I_vector)
    center_of_mass = centerMass(masses, distance_from_center)

    return I_total,center_of_mass

