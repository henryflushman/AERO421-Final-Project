function Xi = xi_matrix(q)
% XI_MATRIX Return quaternion kinematic matrix Xi(q)
% q must be [q1; q2; q3; q4], scalar last

    q = q(:);
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);

    Xi = [ ...
         q4, -q3,  q2;
         q3,  q4, -q1;
        -q2,  q1,  q4;
        -q1, -q2, -q3];
end