function [r_ECI, v_ECI] = coe2rv(coe)
% COE2RV Convert classical orbital elements to ECI position and velocity
%
% Input:
%   coe = [h e Omega i omega nu]
%       h     specific angular momentum [km^2/s]
%       e     eccentricity [-]
%       Omega RAAN [rad]
%       i     inclination [rad]
%       omega argument of perigee [rad]
%       nu    true anomaly [rad]
%
% Output:
%   r_ECI    position in ECI frame [km]
%   v_ECI    velocity in ECI frame [km/s]

    mu = 398600; % km^3/s^2

    h     = coe(1);
    e     = coe(2);
    Omega = coe(3);
    inc   = coe(4);
    omega = coe(5);
    nu    = coe(6);

    % Position and velocity in perifocal frame
    r_pf = (h^2/mu) / (1 + e*cos(nu)) * [cos(nu); sin(nu); 0];
    v_pf = (mu/h) * [-sin(nu); e + cos(nu); 0];

    % Rotation from perifocal to ECI
    R3_Omega = [ cos(Omega) -sin(Omega) 0;
                 sin(Omega)  cos(Omega) 0;
                 0           0          1 ];

    R1_inc = [ 1 0 0;
               0 cos(inc) -sin(inc);
               0 sin(inc)  cos(inc) ];

    R3_omega = [ cos(omega) -sin(omega) 0;
                 sin(omega)  cos(omega) 0;
                 0           0          1 ];

    Q_pX = R3_Omega * R1_inc * R3_omega;

    r_ECI = Q_pX * r_pf;
    v_ECI = Q_pX * v_pf;
end