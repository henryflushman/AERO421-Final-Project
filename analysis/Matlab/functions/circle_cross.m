function q = circle_cross(q1,q2)

epsilon1 = [q1(1),q1(2),q1(3)];
eta1 = q1(4);

epsilon2 = [q2(1),q2(2),q2(3)];
eta2 = q2(4);

epsilon = eta1*epsilon2 + eta2*epsilon1 + skew(eta1)*eta2;
eta = eta1*eta2 - transpose(epsilon1)*epsilon2;

q = [epsilon(1),epsilon(2),epsilon(3),eta];

end