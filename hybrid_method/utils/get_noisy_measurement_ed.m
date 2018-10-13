function z = get_noisy_measurement_ed(track_obj_x, R_ed)
% get noisy measuremnt for ED algorithm

z(1) = track_obj_x(1) + randn(1)*sqrt(R_ed(1,1));
z(2) = track_obj_x(2) + randn(1)*sqrt(R_ed(2,2));
z(3) = pi_to_pi(track_obj_x(3) + randn(1)*sqrt(R_ed(3,3)));
%z(4) = (2-0)*rand;   % randomly generate velocity from 0 to 2 m/s

end