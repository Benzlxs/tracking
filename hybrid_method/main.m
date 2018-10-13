%%
%{
This script is created originally on 11/10/2018; 
version='1.0';

Author: Xuesong(Ben) Li, (benzlee08@gmail.com)
University: UNSW
All rights reserved


Algorithm description:
      The simulation method is a hybrid of expensive detection method and
      cheap segmentation one. Expensive detection method can provide
      observations with accurater position informaiton and more
      information about process model, which are represented with bigger
      noise disturbance, while cheap segmenation method provide more
      nosiy position observations.
      Algorithm starts from tracking one moving object and plots the
      change of covariance deviation
%}



%{
    one branch one check to make bugs free
    1. creating inializaiton map and moving target
    2. showing the single moving target moving well
    3. adding the EDM as measurements to update tracking states
    4. checking the profermance with plotting of deviation of covariance
    5. adding the CSM as mearusments to update tracking states
    6. checking the profermance with plot of deviation
    7. hybridizing EDM and CSM together
    8. checkiing the profermance
    
%}



function []=main(map_dir)
clc, clear;
close all;

rng(0,'twister'); % specify the random number generation pattern, so that bugs can be repeated if there are 
format compact    % format the output display
addpath('./utils');
addpath('../SLAM_MOT/utils');

if nargin == 0
    map_dir = './map.mat'; % or other arguments
end

configfile_hybrid;
fig = figs_inialization(WP, MAP_H, MAP_W);
moving_objs;

dt = DT_CONTROLS;
iwp = 1 ;

% inialize the tracking object state
[X, P] = mot_inialization(track_obj, R_ed, R_cd, R_b, R_v, PATTEN);

while iwp ~= 0
    for j=1:N_track_obj
        [track_obj(j).G, track_obj(j).iwp]= compute_steering(track_obj(j).x, track_obj(j).wp, track_obj(j).iwp, AT_WAYPOINT, track_obj(j).G, RATEG, MAXG, dt);
        if track_obj(j).iwp==0 & track_obj(j).LOOP > 1, track_obj(j).iwp = 1; track_obj(j).LOOP = track_obj(j).LOOP ; end % perform loops: if final waypoint reached, go back to first
        track_obj(j).x= vehicle_model(track_obj(j).x,  track_obj(j).V, track_obj(j).G, WHEELBASE, dt);
        % plots
        xxtt = transformtoglobal(track_obj(j).size, track_obj(j).x);
        set(track_obj(j).H.xt_t1, 'xdata', xxtt(1,:), 'ydata', xxtt(2,:))
    end
    
    % prediction
    [X  P]= process_model_hybrid(X, P, dt, Q_trk);    
    
    
    
end


end

function [X, P] = mot_inialization(track_obj, R_ed, R_cd, R_b, R_v, PATTEN)
%{
    function descriptions;
        this is to inialize states of moving object
    input: 
        track_obj: moving objects
        R_ed: uncertainty of expensive detection method measuremnt
        R_cd: uncertainty of cheap detection method measurement
        R_b:  unceratinty of angle measuremnt in CD
        R_v:  uncertainty of velocity inialization in both methods, for
        there is no measurement on velocity,
    output:
        X: initial state of moving object
        P: initial covariane of moving object
%}
X = [];
P = [];
N_track_obj = length(track_obj);
p = zeros(4, 4);
for i  =1:N_track_obj
    if PATTEN == 0
       % ED 
       z = get_noisy_measurement_ed(track_obj(i).x, R_ed);
       x = [z(1); z(2); z(3); (2-0)*rand];     % velocity are randomly given  
       ind = 1:3;
       p(ind,ind) = R_ed ;
       p(4,4) = R_v;
       X(i,:,:)= x;
       P(i,:,:) = p;
    else
       % CD
       z = get_noisy_measurement_cd(track_obj(i).x, R_cd);
       x = [z(1); z(2); pi_to_pi((2*pi*rand) - pi); (2-0)*rand];  % heading and velocity are randomly given
       ind = 1:2;
       p(ind,ind) = R_cd ;
       p(3,3) = R_b;
       p(4,4) = R_v;
       X(i,:,:) = x;
       P(i,:,:) = p;       
    end
end

end

function fig = figs_inialization(wp, map_h, map_w)
%{
    function descriptions: 
        create the map including waypoints
    inputs:
        wp: way checkpoints
        map_h: height of map
        map_w: widht of map
%}
W = map_w;
H = map_h; 
C = [0.5, 0.5, 0.5];  % color for map

fig = figure;
axis([-W, W, -H, H]);
hold on
wp=wp';
plot(wp(1,:),wp(2,:), 'g')
hold on

xlabel('metres'), ylabel('metres')
set(fig, 'name', 'Hybrid Simulator')
set(fig,'units','points','position',[100,100,800,800])
end
