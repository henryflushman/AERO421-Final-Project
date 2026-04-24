%% Final Simulation Project
% Template for part 3 - Disturbance Modeling
% Aero 421
% Eric Mehiel
% Cal Poly, SLO

clear
close all
clc

%% Part 1 - Mass Properties

% Mass properties for normal operations phase
Bus_m  = 500;
Bus_com = [0;0;0];

Sensor_m = 100;
Sensor_com = [0;0;1.5];

Sp_minus_m = 20;
Sp_minus_com = [0;-2.5;0];

Sp_plus_m = 20;
Sp_plus_com = [0;2.5;0];

% Function to calculate I of rectangular prism 
function I = rect_prism_I(m,x,y,z)
    Ixx = (1/12)*m*(y^2+z^2);
    Iyy = (1/12)*m*(x^2+z^2);
    Izz = (1/12)*m*(x^2+y^2);

    I = [Ixx 0 0; 0 Iyy 0; 0 0 Izz];
end

% Calculates individual Inertia Matrices 
Bus_I = rect_prism_I(Bus_m, 2,2,2);

Sensor_I  = rect_prism_I(Sensor_m,0.25, 0.25,1);

Sp_minus_I = rect_prism_I(Sp_minus_m,2,3,0.05);

Sp_plus_I = rect_prism_I(Sp_plus_m,2,3,0.05);

% Total Mass 
total_mass =  Bus_m + Sensor_m + Sp_minus_m + Sp_plus_m;

% Total COM
com = (Bus_com*Bus_m + Sensor_com*Sensor_m + Sp_minus_com*Sp_minus_m + Sp_plus_com*Sp_plus_m)/total_mass;

% a^X skew matrix function
function X = skew(vec)
X = [0 -vec(3) vec(2); vec(3) 0 -vec(1); -vec(2) vec(1) 0];
end

% Defines vector from individual component com to s/c com 
r_Sensor = Sensor_com - com;
r_Sp_minus = Sp_minus_com - com;
r_Sp_plus = Sp_plus_com - com;
r_Bus = Bus_com - com;

% Normal Operation Interia Matrix 
J = (Bus_I - Bus_m*skew(r_Bus)*skew(r_Bus)) +...
    (Sensor_I - Sensor_m*skew(r_Sensor)*skew(r_Sensor)) + ...
    (Sp_minus_I - Sp_minus_m*skew(r_Sp_minus)*skew(r_Sp_minus)) +...
    (Sp_plus_I - Sp_plus_m*skew(r_Sp_plus)*skew(r_Sp_plus));


% Print 
fprintf('The spacecraft mass for the normal operations mode is:\n')
display(total_mass)
fprintf('The spacecraft center of mass for the normal operations mode is:\n')
display(com)
fprintf('The Inertia Matrix for the normal operations mode is:\n')
display(J)

%% Part 2 - Geometric Properties of MehielSat during Normal Operations

% Define the sun in F_ECI and residual dipole moment in F_b
Sun_vec = [1;0;0]; % vernal equinox 

residual_dipole_moment = [0;0;-0.5];

% I constructed a matrix where the rows represent each surface of the
% MehielSat.  The first column stores the Aera of the surface, the next
% three columns define the normal vector of that surface in F_b, and the
% final three columns store the center of presure of the surface (geometric
% center of the surface) in F_b.

% First get rhos vectors with respect to the center of the spacecraft bus
% the MehielSat BUS is a box
Areas = 4*ones(6,1);
normals = [];
cps = [];
normals = [normals;1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
cps = [cps;1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];

% Append geometric properties for Solar Panel 1
Areas = [Areas;0.05*3;0.05*3;0.05*2;3*2;3*2];
normals = [normals;1 0 0; -1 0 0; 0 -1 0; 0 0 1; 0 0 -1];
cps = [cps;1,-2.5,0;-1,-2.5,0;0,-4,0;0,-2.5,0.025; 0,-2.5,-0.025 ];
% Append geometric properties for Solar Panel 2
Areas = [Areas;0.05*3;0.05*3;0.05*2;3*2;3*2];
normals = [normals;1 0 0; -1 0 0; 0 1 0;  0 0 1; 0 0 -1];
cps = [cps;1,2.5,0;-1,2.5,0;0,4,0;0,2.5,0.025; 0,2.5,-0.025];
% Append geometric properties for Sensor
Areas = [Areas;0.25;0.25;0.25;0.25;0.125 ];
normals = [normals;1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 -1];
cps = [cps;
    0.125, 0, 1.5;
    -0.125, 0, 1.5;
    0, 0.125, 1.5;
    0, -0.125, 1.5;
    0, 0, 2 ];
% now subtract the center of mass to get the location of the rho vectors
% with respect to the center of mass
rhos = cps-com.';
% Now build the matrix
surfaceProperties = [Areas cps normals];

%% Part 3 - Initialize Simulation States

% Current JD - has to be on the solar equinox, why? - we'll use 3/20/2024
% from https://aa.usno.navy.mil/data/JulianDate
% Need this so we can convert from F_ECEF to F_ECI and to F_b for the
% magnetic field model
JD_0 = 2460390;

% Spacecraft Orbit Properties
mu = 398600; % km^3/s^2
h = 53335.2; % km^2/s
e = 0; % none
Omega = 0*pi/180; % radians
inclination = 98.43*pi/180; % radians
omega = 0*pi/180; % radians
nu = 0*pi/180; % radians

a = h^2/mu/(1 - e^2);
orbital_period = 2*pi*sqrt(a^3/mu);

% Set/Compute initial conditions
% intial orbital position and velocity
[r_ECI_0, v_ECI_0] = coe2rv([h e Omega inclination omega nu]);

% No external command Torque
T_c = [0; 0; 0]; % Nm

% Compute initial F_LVLH basis vectors in F_ECI components
rhat = r_ECI_0 / norm(r_ECI_0);
hvec = cross(r_ECI_0, v_ECI_0);
hhat = hvec / norm(hvec);
that = cross(-rhat, hhat);
that = that / norm(that);

% Initial Euler angles relating F_body and F_LVLH (given)
phi_0 = 0;
theta_0 = 0;
psi_0 = 0;
E_b_LVLH_0 = [phi_0; theta_0; psi_0];

% Initial Quaternion relating F_body and F_LVLH (given)
q_b_LVLH_0 = [0; 0; 0; 1];

% DCM from ECI to LVLH
C_LVLH_ECI_0 = [hhat.';
                that.';
               (-rhat).'];

% DCM from LVLH to body
c1 = cos(phi_0);
s1 = sin(phi_0);
c2 = cos(theta_0);
s2 = sin(theta_0);
c3 = cos(psi_0);
s3 = sin(psi_0);

C_b_LVLH_0 = [c2*c3,                c2*s3,               -s2;
              s1*s2*c3-c1*s3,       s1*s2*s3+c1*c3,      s1*c2;
              c1*s2*c3+s1*s3,       c1*s2*s3-s1*c3,      c1*c2];

C_b_ECI_0 = C_b_LVLH_0 * C_LVLH_ECI_0;

% Initial Euler angles relating body to ECI
E_b_ECI_0 = C2EulerAngles(C_b_ECI_0);

% Initial quaternion relating body to ECI
q_b_ECI_0 = C2quat(C_b_ECI_0);

% Initial body rates of spacecraft (given)
w_b_ECI_0 = [0.001; -0.001; 0.002];

%% Part 4 - Simulate Results

n_revs = 5; %revs
tspan = n_revs * orbital_period;
out = sim('FP_Solutions_Part_3_disturbance');

%% Part 5 - Plot Results

% Plot Angular Velocities, Euler Angles and Quaternions

% Plot Disturbance torques in F_b
