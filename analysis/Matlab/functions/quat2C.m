function C = quat2C(q)
% QUAT2C Convert quaternion [q1; q2; q3; q4] to DCM
% Scalar is LAST.
%
% Input:
%   q : 4x1 quaternion [q1; q2; q3; q4]
%
% Output:
%   C : 3x3 direction cosine matrix
%
% Based on quaternion_to_dcm() from SatelliteObject.py

    q = q(:) / norm(q);
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);

    C = [ ...
        q4^2 + q1^2 - q2^2 - q3^2,   2*(q1*q2 - q3*q4),             2*(q1*q3 + q2*q4);
        2*(q1*q2 + q3*q4),           q4^2 - q1^2 + q2^2 - q3^2,    2*(q2*q3 - q1*q4);
        2*(q1*q3 - q2*q4),           2*(q2*q3 + q1*q4),            q4^2 - q1^2 - q2^2 + q3^2];
end