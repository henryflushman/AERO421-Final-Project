function q = C2quat(C)
% C2QUAT Convert DCM to quaternion [q1; q2; q3; q4]
% Scalar is LAST.
%
% Input:
%   C : 3x3 direction cosine matrix
%
% Output:
%   q : 4x1 quaternion [q1; q2; q3; q4]
%
% Based on dcm_to_quaternion() from SatelliteObject.py

    % Compute all four components
    q1 = sqrt(max(0, 0.25 * (1.0 + C(1,1) - C(2,2) - C(3,3))));
    q2 = sqrt(max(0, 0.25 * (1.0 - C(1,1) + C(2,2) - C(3,3))));
    q3 = sqrt(max(0, 0.25 * (1.0 - C(1,1) - C(2,2) + C(3,3))));
    q4 = sqrt(max(0, 0.25 * (1.0 + C(1,1) + C(2,2) + C(3,3))));
    
    % Always use q1-based formula (matches Python implementation)
    q = (1/(4*q1)) * [ ...
        4*q1^2;
        C(1,2) + C(2,1);
        C(3,1) + C(1,3);
        C(2,3) - C(3,2)];
    
    % Normalize
    q = q / norm(q);
    
    % Apply canonical form: scalar >= 0
    if q(4) < 0
        q = -q;
    end
end