function [z_trk_obj, ind_trk_obj]= get_observations_tracking_obj(x, trk_obj, tag_trk_obj, rmax)
%
% INPUTS:
%   x       - vehicle pose [x;y;phi]
%   trk_obj - set of all moving objects
%   tag_trk_obj - index tags for each moving objects
%   rmax - maximum range of range-bearing sensor 
%
% OUTPUTS:
%   z - set of range-bearing observations
%   ind_trk_obj - moving object index for each observation
%
% Xuesong LI, 2018.

[state_obj,ind_trk_obj]= get_visible_moving_object(x, trk_obj, tag_trk_obj,rmax);
z_trk_obj= compute_range_bearing(x,state_obj);

%
%

function [state_obj, ind_trk_obj]= get_visible_moving_object(x, trk_obj, tag_trk_obj, rmax)
% Select set of moving objects that are visible within vehicle's semi-circular field-of-view
all_states_trk_obj = [trk_obj.x];
dx = all_states_trk_obj(1,:) - x(1);
dy = all_states_trk_obj(2,:) - x(2);
phi= x(3);

% incremental tests for bounding semi-circle
ii= find(abs(dx) < rmax & abs(dy) < rmax ... % bounding box
      & (dx*cos(phi) + dy*sin(phi)) > 0 ...  % bounding line
      & (dx.^2 + dy.^2) < rmax^2);           % bounding circle
% Note: the bounding box test is unnecessary but illustrates a possible speedup technique
% as it quickly eliminates distant points. Ordering the landmark set would make this operation
% O(logN) rather that O(N).
  
state_obj = all_states_trk_obj(:,ii);
ind_trk_obj= tag_trk_obj(ii);

%
%

function z_trk_obj= compute_range_bearing(x, state_obj)
% Compute exact observation
dx= state_obj(1,:) - x(1);
dy= state_obj(2,:) - x(2);
phi= x(3);
z_trk_obj= [sqrt(dx.^2 + dy.^2);
    atan2(dy,dx) - phi];
    