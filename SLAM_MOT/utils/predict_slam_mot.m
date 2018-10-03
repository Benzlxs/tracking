function [x,P]= predict_slam_mot(x,P,v,g,Q,WB,dt, num_lm, Q_MOT)
%function [xn,Pn]= predict (x,P,v,g,Q,WB,dt)
%
% Inputs:
%   x, P - SLAM state and covariance
%   v, g - control inputs: velocity and gamma (steer angle)
%   Q - covariance matrix for velocity and gamma
%   WB - vehicle wheelbase
%   dt - timestep
%   num_lm - number of landmarks
%   Q_MOT - the uncertainties of moving object 
%
% Outputs: 
%   xn, Pn - predicted state and covariance
%
% Xuesong LI, 2018

n = size(x,1);
J =  eye(n);

% jacobians for robots
s= sin(g+x(3)); c= cos(g+x(3));
vts= v*dt*s; vtc= v*dt*c;  

J(1:3,1:3)= [1 0 -vts;
             0 1  vtc;
             0 0  1];
Gu= [dt*c          -vts;
      dt*s          vtc;
      dt*sin(g)/WB  v*dt*cos(g)/WB];

% jacobians for moving objects
assert(mod((size(x,1) - 3 - num_lm*2),4) == 0,'number of moving objects should be integer');
num_mot = (size(x,1) - 3 - num_lm*2)/4;

% predict state
x(1:3)= [x(1) + vtc; 
         x(2) + vts;
         pi_to_pi(x(3)+ v*dt*sin(g)/WB)];


for i = 1:num_mot
    ind = 3 + num_lm*2 + 4*i- 1;
    s= sin( x(ind)); c= cos( x(ind));
    vts= x(ind+1)*dt*s; vtc= x(ind+1)*dt*c;
    ct = c*dt; st = dt*s;
    J(ind-2:ind+1,ind-2:ind+1) = [1 0 -vts ct;
                                  0 1  vtc st;
                                  0 0   1   0;
                                  0 0   0   1];
end

P = J*P*J';
 % adding the input noise
P(1:3,1:3)= P(1:3,1:3) + Gu*Q*Gu';

% adding motion model noise
for i = 1:num_mot
    ind = 3 + num_lm*2 + 4*i- 1;
    s= sin( x(ind)); c= cos( x(ind));
    vts= x(ind+1)*dt*s; vtc= x(ind+1)*dt*c;
    % adding the motion noise
    P(ind-2:ind+1,ind-2:ind+1) =  P(ind-2:ind+1,ind-2:ind+1) + dt^2*Q_MOT;
    % predict state
    x(ind-2:ind+1)  = [x(ind-2) + vtc; 
                       x(ind-1) + vts;
                       pi_to_pi(x(ind ));
                       abs(x(ind+1))];
end

