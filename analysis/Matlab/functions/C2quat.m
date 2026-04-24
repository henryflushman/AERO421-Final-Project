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

    q1 = sqrt(max(0, 0.25 * (1.0 + C(1,1) - C(2,2) - C(3,3))));
    q2 = sqrt(max(0, 0.25 * (1.0 - C(1,1) + C(2,2) - C(3,3))));
    q3 = sqrt(max(0, 0.25 * (1.0 - C(1,1) - C(2,2) + C(3,3))));
    q4 = sqrt(max(0, 0.25 * (1.0 + C(1,1) + C(2,2) + C(3,3))));

    if q1 < 1e-12
        % Fallback branch if q1 is too small
        [~, idx] = max([q1 q2 q3 q4]);

        switch idx
            case 1
                q = (1/(4*q1)) * [ ...
                    4*q1^2;
                    C(1,2) + C(2,1);
                    C(3,1) + C(1,3);
                    C(2,3) - C(3,2)];
            case 2
                q = (1/(4*q2)) * [ ...
                    C(1,2) + C(2,1);
                    4*q2^2;
                    C(2,3) + C(3,2);
                    C(3,1) - C(1,3)];
            case 3
                q = (1/(4*q3)) * [ ...
                    C(3,1) + C(1,3);
                    C(2,3) + C(3,2);
                    4*q3^2;
                    C(1,2) - C(2,1)];
            otherwise
                q = (1/(4*q4)) * [ ...
                    C(2,3) - C(3,2);
                    C(3,1) - C(1,3);
                    C(1,2) - C(2,1);
                    4*q4^2];
        end
    else
        q = (1/(4*q1)) * [ ...
            4*q1^2;
            C(1,2) + C(2,1);
            C(3,1) + C(1,3);
            C(2,3) - C(3,2)];
    end

    if q(4) < 0
        q = -q;
    end

    q = q / norm(q);
end