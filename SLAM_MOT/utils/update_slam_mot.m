function [x,P]= update_slam_mot( x, P, z_lm, R_lm, idf_lm, zf_mot, R_mot, idf_mot, batch, num_lm)
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

lenz= size(z_lm,2) + size(zf_mot, 2);  % getting all observation
num_z_lm = size(z_lm,2);
lenx= length(x);
H= zeros(2*lenz, lenx);
v= zeros(2*lenz, 1);
RR= zeros(2*lenz);

% getting observation model for landmark observation
for i=1:num_z_lm
    ii= 2*i + (-1:0);
    [zp,H(ii,:)]= observe_model(x, idf_lm(i));
    
    v(ii)=      [z_lm(1,i)-zp(1);
        pi_to_pi(z_lm(2,i)-zp(2))];
    RR(ii,ii)= R_lm;
end
% getting the observation model for moving targets
for i = (num_z_lm+1):lenz
    ii= 2*i + (-1:0);
    iidf = i - num_z_lm;
    [zp,H(ii,:)]= observe_model_slam_mot(x, idf_mot(iidf), num_lm);
    
    v(ii)=      [zf_mot(1,iidf)-zp(1);
        pi_to_pi(zf_mot(2,iidf)-zp(2))];
    RR(ii,ii)= R_mot;
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
