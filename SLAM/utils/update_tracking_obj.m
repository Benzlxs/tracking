function [obj_x , obj_P]= update_tracking_obj(rob_x, obj_x , obj_P, z, R, idf)
% function [x,P]= update(x,P,z,R,idf, batch)
%
% Inputs:
%   x, P - state and covariance of tracked moving objects
%   z, R - range-bearing measurements and covariances
%   idf - feature index for each z
%
% Outputs:
%   x, P - updated state and covariance

 
% [x,P]= batch_update(x,P,z,R,idf);
 [obj_x , obj_P]= single_update(rob_x, obj_x, obj_P, z, R, idf); %single updating for every tracked object 

%
% 
% function [x,P]= batch_update(x,P,z,R,idf)
% 
% lenz= size(z,2);
% lenx= length(x);
% H= zeros(2*lenz, lenx);
% v= zeros(2*lenz, 1);
% RR= zeros(2*lenz);
% 
% for i=1:lenz
%     ii= 2*i + (-1:0);
%     [zp,H(ii,:)]= observe_model(x, idf(i));
%     
%     v(ii)=      [z(1,i)-zp(1);
%         pi_to_pi(z(2,i)-zp(2))];
%     RR(ii,ii)= R;
% end
%         
% [x,P]= KF_cholesky_update(x,P,v,RR,H);
%
%

function [obj_x,obj_P]= single_update(rob_x, obj_x, obj_P, z, R, idf)

lenz= size(z,2);
for i=1:lenz
    [zp,H]= observe_model_tracking_obj(rob_x , obj_x(idf(i),:));
    
    v= [z(1,i)-zp(1);
        pi_to_pi(z(2,i)-zp(2))];
    p_i = squeeze(obj_P(idf(i),:,:));
    [x,P]= KF_cholesky_update(obj_x(idf(i),:)' , p_i,v, R,H);
    obj_x(idf(i),:) = x';
    obj_P(idf(i),:,:) = P;
end        
