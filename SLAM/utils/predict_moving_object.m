function [x,P]= predict_moving_object (x,P,Q,dt)
%function [xn,Pn]= predict (x,P,v,g,Q,WB,dt)
%
% Inputs:
%   x, P - moving object state and covariance
%   Q    - noise of processing model
%   dt - timestep
%
% Outputs: 
%   xn, Pn - predicted state and covariance
%
% Xuesong LI, 2018

s= sin( x(3)); c= cos( x(3));
vts= x(4)*dt*s; vtc= x(4)*dt*c;
ct = c*dt; st = dt*s;
% jacobians   
Gv= [1 0 -vts ct;
     0 1  vtc st;
     0 0   1   0;
     0 0   0   1];
  
% predict covariance

P = Gv*P*Gv' +  dt^2*Q;  % there is no noise from process model, assume that process model is correct, not approximation

% predict state
x  = [x(1) + vtc; 
      x(2) + vts;
         x(3);
         x(4)];

