function [x , P, count_trk, ind_trk_obj]= augment_slam_mot_mot(x,P, zn_mot, zn_ind, R_trk, num_lm, ind_trk_obj, count_trk)
%function [x_trk , P_trk, count_trk, ind_trk_obj]= augment_slam_mot_mot(x_rob, P_rob, x_trk , P_trk, zn_trk, RE, count_trk, zn_ind, ind_trk_obj)
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
zn_num = size(zn_mot,2);
for i=1:zn_num
    [x , P]= add_one_z(x, P , zn_mot(:,i), R_trk);
    count_trk = [count_trk; 0];
    ind_trk_obj = [ind_trk_obj; zn_ind(i)];
end

%
%

function [x , P]= add_one_z(x, P, zn, R)
% format of x_trk is n*m, and one of P_trk is n*m*m
len= size(x, 1);
r= zn(1); b= zn(2);
s= sin(x(3)+b); 
c= cos(x(3)+b);

% augment x
x_temp = [x(1) + r*c;
          x(2) + r*s;
          pi_to_pi(x(3)+b+pi);
          1 ]; % direction of car is negative of bearing
x= [x;  x_temp];
pos = size(x, 1);

% jacobians, uncertainty comes from robot's uncertainty and observation's
Gv= [1  0 -r*s;
     0  1  r*c;
     0  0   1;
     0  0   0];

Gz= [c  -r*s;
     s   r*c;
     0    1  ;
     0    0  ];
     
% augment P
P_temp= Gv*P(1:3,1:3)*Gv' + Gz*R*Gz'; % feature cov

rng = (len+1):(len+4) ; 
P(rng, rng) = P_temp; 
P(rng,1:3) = Gv*P(1:3,1:3);
P(1:3,rng)= P(rng,1:3)';
rnm = 4:len;
P(rng, rnm) = 0; % assuming that landmark and moving object are independent
P(rnm, rng) = 0;


