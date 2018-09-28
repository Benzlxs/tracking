function [x_trk , P_trk, count_trk, ind_trk_obj]= augment_tracking_obj(x_rob, P_rob, x_trk , P_trk, zn_trk, RE, count_trk, zn_ind, ind_trk_obj)
%function [x,P]= augment(x,P,z,R)
%
% Inputs:
%   x_trk, P_trk  - state and covariance of tracked objects
%   zn_trk, RE  - range-bearing measurements and covariances, each of a new feature
%   zn_ind   - index to object in augmentation observation
%   ind_trk_obj  -- index in tracking state set
%
% Outputs:
%   x_trk , P_trk - augmented state and covariance
%   count_trk, ind_trk_obj: counter and index to real moving object
%
% Notes: 
%   - We assume the number of moving object is foure.
%   - Only one value for R is used, as all measurements are assumed to 
%   have same noise properties.
%
% Xuesong Li, 2018.

% add new features to state
zn_num = size(zn_trk,2);
for i=1:zn_num
    [x_trk , P_trk]= add_one_z(x_rob, P_rob , x_trk , P_trk, zn_trk(:,i), RE);
    count_trk = [count_trk; 0];
    ind_trk_obj = [ind_trk_obj; zn_ind(i)];
end

%
%

function [x_trk , P_trk]= add_one_z(x_rob, P_rob , x_trk, P_trk, zn, R)
% format of x_trk is n*m, and one of P_trk is n*m*m
len= size(x_trk, 2);
r= zn(1); b= zn(2);
s= sin(x_rob(3)+b); 
c= cos(x_rob(3)+b);

% augment x
x_temp = [x_rob(1) + r*c,  x_rob(2) + r*s,  pi_to_pi(x_rob(3)+b+pi),  1]; % direction of car is negative of bearing
x_trk= [x_trk;  x_temp];
pos = size(x_trk, 1);

% jacobians, uncertainty comes from robot's uncertainty and observation's
Gv= [1  0 -r*s;
     0  1  r*c;
     0  0   0;
     0  0   0];
 
     
% augment P
P_temp= Gv*P_rob*Gv'; % feature cov
P_temp(3:4,3:4) = R;

P_trk(pos,:,:) = P_temp;


