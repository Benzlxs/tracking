function [X P] = update_ed(X, P, z, R)
%{
    Using Kalman filter to up the state
    input:
        x: state of tracked robot 4*1
        p: covariance of tracked robot 4*4
        z: observation, 3*1, [x y alpha]
        R_ed: noise of observation
    output:
        x: updated state
        p: updated covariance
%}

H = [1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, 0;];

v = z-X(1:3);
v(end) = pi_to_pi(v(end));

PHt= P*H';
S= H*PHt + R;

S= (S+S')*0.5; % make symmetric
SChol= chol(S);

SCholInv= inv(SChol); % triangular matrix
W1= PHt * SCholInv;
W= W1 * SCholInv';

X= X' + W*v'; % update 
P= P - W1*W1';
end