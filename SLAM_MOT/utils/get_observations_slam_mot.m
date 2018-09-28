function [z_lm, idf, z_mot_obj, ind_mot_z]= get_observations_slam_mot(x, lm, idf, rmax, num_lm, trk_obj, tag_trk_obj)
%function [z,idf]= get_observations(x, lm, idf, rmax)
%
% INPUTS:
%   x - vehicle pose [x;y;phi]
%   lm - set of all landmarks
%   idf - index tags for each landmark
%   rmax - maximum range of range-bearing sensor 
%   num_lm - the number of land marks
%
% OUTPUTS:
%   z - set of range-bearing observations
%   idf - landmark index tag for each observation
%
% Tim Bailey 2004.
[lm,idf]= get_visible_landmarks(x,lm,idf,rmax);
z_lm= compute_range_bearing(x,lm);

[state_obj, ind_mot_z]= get_visible_moving_object(x, trk_obj, tag_trk_obj,rmax);
z_mot_obj= compute_range_bearing_moving_object(x,state_obj);

%
%

function z_trk_obj= compute_range_bearing_moving_object(x, state_obj)
% Compute exact observation
dx= state_obj(1,:) - x(1);
dy= state_obj(2,:) - x(2);
phi= x(3);
z_trk_obj= [sqrt(dx.^2 + dy.^2);
    atan2(dy,dx) - phi];
    
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
function [lm,idf]= get_visible_landmarks(x,lm,idf,rmax)
% Select set of landmarks that are visible within vehicle's semi-circular field-of-view
dx= lm(1,:) - x(1);
dy= lm(2,:) - x(2);
phi= x(3);

% incremental tests for bounding semi-circle
ii= find(abs(dx) < rmax & abs(dy) < rmax ... % bounding box
      & (dx*cos(phi) + dy*sin(phi)) > 0 ...  % bounding line
      & (dx.^2 + dy.^2) < rmax^2);           % bounding circle
% Note: the bounding box test is unnecessary but illustrates a possible speedup technique
% as it quickly eliminates distant points. Ordering the landmark set would make this operation
% O(logN) rather that O(N).
  
lm= lm(:,ii);
idf= idf(ii);

%
%

function z= compute_range_bearing(x,lm)
% Compute exact observation
dx= lm(1,:) - x(1);
dy= lm(2,:) - x(2);
phi= x(3);
z= [sqrt(dx.^2 + dy.^2);
    atan2(dy,dx) - phi];
    