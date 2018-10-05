function [x,P]= augment_slam_mot_lm(x,P,z,R, num_lm)
%function [x,P]= augment(x,P,z,R)
%
% Inputs:
%   x, P - SLAM state and covariance
%   z, R - range-bearing measurements and covariances, each of a new feature
%
% Outputs:
%   x, P - augmented state and covariance
%
% Notes: 
%   - We assume the number of vehicle pose states is three.
%   - Only one value for R is used, as all measurements are assumed to 
%   have same noise properties.
%


% add new features to state
for i=1:size(z,2)
    ind =3 + num_lm*2 + 2*(i-1);
    [x,P]= add_one_z(x,P,z(:,i),R, ind);
end

%
%

function [x,P]= add_one_z(x,P_old,z,R, ind)

len= ind;
all_len = length(x);
r= z(1); b= z(2);
s= sin(x(3)+b); 
c= cos(x(3)+b);

% augment x
x= [x(1:ind);
    x(1) + r*c;
    x(2) + r*s;
    x(ind+1:end)];
P = zeros(length(x), length(x));
P(1:ind, 1:ind) = P_old(1:ind,1:ind); % keep the previous covariance
% jacobians
Gv= [1 0 -r*s;
     0 1  r*c];
Gz= [c -r*s;
     s  r*c];
     
% augment P
rng= len+1:len+2;
P(rng,rng)= Gv*P(1:3,1:3)*Gv' + Gz*R*Gz'; % feature cov
P(rng,1:3)= Gv*P(1:3,1:3); % vehicle to feature xcorr
P(1:3,rng)= P(rng,1:3)';
if len>3
    rnm= 4:len;
    P(rng,rnm)= Gv*P(1:3,rnm); % map to feature xcorr
    P(rnm,rng)= P(rng,rnm)';
end

% there is moving objects
if all_len > ind
    rnm = (ind+1):all_len;
    rnm_new = 2+rnm;
    P(rng, rnm_new) = Gv*P_old(1:3,rnm);
    P(rnm_new, rng) = P(rng, rnm_new)';
end

P(ind+3:end, ind+3:end) = P_old(ind+1:end, ind+1:end);
P(ind+3:end, 1:ind) = P_old(ind+1:end, 1:ind);
P(1:ind, ind+3:end) = P_old(1:ind, ind+1:end);