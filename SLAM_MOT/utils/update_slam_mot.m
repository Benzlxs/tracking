function [x,P]= update_slam_mot(x,P, z_lm, R_lm, idf_lm, zf_mot, R_mot, idf_mot, batch, num_lm)
% Inputs:
%   x, P - SLAM state and covariance
%   z_lm , R_lm   - range-bearing measurements and covariances of landmarks
%   zf_mot, R_mot - range-bearing measurements and covariances of moving
%   objects
%   idf_lm - feature index for landmark
%   idf_mot - feature index for each moving object
%   batch - switch to specify whether to process measurements together or sequentially
%   num_lm - number of landmarks
%
% Outputs:
%   x, P - updated state and covariance

[x,P]= batch_update(x, P, z_lm, R_lm, idf_lm, zf_mot, R_mot, idf_mot, num_lm);

%
%

function [x,P]= batch_update(x, P, z_lm, R_lm, idf_lm, zf_mot, R_mot, idf_mot, num_lm)

lenz= size(z,2);
lenx= length(x);
H= zeros(2*lenz, lenx);
v= zeros(2*lenz, 1);
RR= zeros(2*lenz);

for i=1:lenz
    ii= 2*i + (-1:0);
    [zp,H(ii,:)]= observe_model(x, idf(i));
    
    v(ii)=      [z(1,i)-zp(1);
        pi_to_pi(z(2,i)-zp(2))];
    RR(ii,ii)= R;
end
        
[x,P]= KF_cholesky_update(x,P,v,RR,H);

%
%

function [x,P]= single_update(x,P,z,R,idf)

lenz= size(z,2);
for i=1:lenz
    [zp,H]= observe_model(x, idf(i));
    
    v= [z(1,i)-zp(1);
        pi_to_pi(z(2,i)-zp(2))];
    
    [x,P]= KF_cholesky_update(x,P,v,RR,H);
end        
