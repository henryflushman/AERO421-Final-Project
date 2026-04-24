function E = C2EulerAngles(C, sequence)
% C2EULERANGLES Convert DCM to Euler angles (no toolboxes required)
%
% Usage:
%   E = C2EulerAngles(C)
%   E = C2EulerAngles(C, '321')
%
% Default is 3-2-1 sequence (roll, pitch, yaw)
%
% Output:
%   E = [phi; theta; psi]

    if nargin < 2
        sequence = '321';
    end

    sequence = upper(sequence);

    switch sequence
        case {'321','ZYX'}
            % 3-2-1 (yaw-pitch-roll)
            theta = asin(-C(1,3));
            psi   = atan2(C(1,2), C(1,1));
            phi   = atan2(C(2,3), C(3,3));

            E = [phi; theta; psi];

        case {'313','ZXZ'}
            % Optional: symmetric sequence if needed later
            theta = acos(C(3,3));
            psi   = atan2(C(3,1), -C(3,2));
            phi   = atan2(C(1,3), C(2,3));

            E = [phi; theta; psi];

        otherwise
            error('Unsupported sequence. Use ''321'' or ''313''.');
    end
end