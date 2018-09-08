function [z,H]= observe_model_tracking_obj(rob_x, obj_x)
%function [z,H]= observe_model(x, idf)
%
% INPUTS:
%   obj_x - state vector of tracked objects
%   idf - index of tracked object
%   rob_x - state vector of robot ego
%
% OUTPUTS:
%   z - predicted observation
%   H - observation Jacobian
%
% Given a feature index (ie, the order of the feature in the state vector),
% predict the expected range-bearing observation of this feature and its Jacobian.
%
% Xuesong LI 2018.

%Nxv= 4; % number of every tracked object
Ns = 4;  % number of state of tracked object
%fpos= (idf-1)*Nxv +1; % position of xf in state
%H= zeros(2, length(obj_x));
H= zeros(2, Ns);

% auxiliary values
% dx= obj_x(fpos)  -rob_x(1); 
% dy= obj_x(fpos+1)-rob_x(2);
dx= rob_x(1) - obj_x(1); % rob_x(1) is constant, obj_x is the state
dy= rob_x(2) - obj_x(2);
d2= dx^2 + dy^2;
d= sqrt(d2);
xd = dx/d ;
yd = dy/d ;
xd2= dx/d2;
yd2= dy/d2;

% predict z
z= [d;
    atan2(dy,dx) - rob_x(3)];

% calculate H
H(:,1:Ns)  = [-xd -yd 0 0; yd2 -xd2 -1 0];

