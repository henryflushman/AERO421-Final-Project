function S = skew(v)
% SKEW Return 3x3 skew-symmetric matrix such that skew(a)*b = cross(a,b)

    v = v(:);

    S = [ ...
         0    -v(3)  v(2);
         v(3)  0    -v(1);
        -v(2)  v(1)  0   ];
end