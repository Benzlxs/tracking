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
h = figs_inialization(WP, MAP_H, MAP_W);
moving_objs;

dt = DT_CONTROLS;
iwp = 1 ;

% inialize the tracking object state
[X, P, data] = mot_inialization(track_obj, R_ed, R_cd, R_b, R_v, PATTEN);

dtsum=0;
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
    if PATTEN == 0
        [X  P]= process_model_hybrid(X, P, dt, Q_trk_ed);    
    else
        [X  P]= process_model_hybrid(X, P, dt, Q_trk_cd);  
    end
    
    dtsum= dtsum + dt;
    % observation to update
    if dtsum >= DT_OBSERVE
        dtsum= 0;
        for i  =1:N_track_obj            
            if PATTEN == 0                                                           
                % ED 
                % get noisy measuremnt
                z = get_noisy_measurement_ed(track_obj(i).x, R_ed);
                % Kalman filte updating
                [x p] = update_ed(X(i,:), squeeze(P(i,:,:)), z, R_ed);
                X(i,:) = x;
                P(i,:,:) = p;
            else
                % CD
                z = get_noisy_measurement_cd(track_obj(i).x, R_cd);
                % Kalman filte updating
                [x p] = update_cd(X(i,:), squeeze(P(i,:,:)), z, R_cd);
                X(i,:) = x;
                P(i,:,:) = p;
            end
        end
    end
    %% plotting
    % offline data store
    plines = [];
    for i =1:N_track_obj
        %TO-DO: predict moving object one by one
        x_obj=transformtoglobal(track_obj(i).size, X(i,:));
        %set(track_obj(ij).H.xv_t1,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        set( fig_hs(i).car,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        p_conv= make_covariance_ellipses_tracking_obj(X(i,:), squeeze(P(i,:,:)));
        %set(track_obj(ij).H.cov_t1,'xdata', p_conv(1,:), 'ydata', p_conv(2,:));
        set( fig_hs(i).elliphse,'xdata',p_conv(1,:), 'ydata', p_conv(2,:));
        % set
        data= store_data(data, X(i,:), squeeze(P(i,:,:)), track_obj(i).x, i);
        
        % plotting the how difference and uncertainty change with time
        set( fig_dif(i).x,'xdata', 1:data(j).i, 'ydata',  data(j).error(1:data(j).i));
        set( fig_dif(i).p,'xdata', 1:data(j).i, 'ydata',  data(j).uncertain(1:data(j).i));
        set(  h.obs, 'xdata', [0 X(i,1)], 'ydata',[0 X(i,2)] )
        
    end
    pause(0.05);
    
   
end
end

%
%

function data= store_data(data, x, P, xtrue, j)
% add current data to offline storage
CHUNK= 5000;
if data(j).i == size(data(j).path,2) % grow array in chunks to amortise reallocation
    data(j).path= [data(j).path zeros(3,CHUNK)];
    data(j).true= [data(j).true zeros(3,CHUNK)];
end
i= data(j).i + 1;
data(j).i= i;
data(j).path(:,i)= x(1:3);
data(j).true(:,i)= xtrue;
data(j).state(i).x= x;
%data.state(i).P= P;
data(j).state(i).P= diag(P);
data(j).error(i) = mean(abs(x(1:3)' - xtrue));
data(j).uncertain(i) = mean(sqrt(diag(P)));

end
%
%
function p= make_covariance_ellipses_tracking_obj(x,P)
% compute ellipses for plotting state covariances
N= 10;
inc= 2*pi/N;
phi= 0:inc:2*pi;


p= zeros (2,1*(N+2));

ii=1:N+2;
p(:,ii)= make_ellipse(x(1:2), P(1:2,1:2), 2, phi);

end
%
%
function p= make_ellipse(x,P,s, phi)
% make a single 2-D ellipse of s-sigmas over phi angle intervals 
r= sqrtm(P);
a= s*r*[cos(phi); sin(phi)];
p(2,:)= [a(2,:)+x(2) NaN];
p(1,:)= [a(1,:)+x(1) NaN];
end
%
%

function [X, P, data] = mot_inialization(track_obj, R_ed, R_cd, R_b, R_v, PATTEN)
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
data = [];
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
    data= initialise_store(data,x(1:3),p,x(1:3),i); % stored data for off-line
end

end

%
% Initial data
function data= initialise_store(data,x,P, xtrue,i)
% offline storage initialisation
data(i).i=1;
data(i).path= x;
data(i).true= xtrue;
data(i).state(1).x= x;
%data.state(1).P= P;
data(i).state(1).P= diag(P);
data(i).error(i) = mean(abs(x(1:3) - xtrue));
data(i).uncertain(i) = mean(sqrt(diag(P)));
end
%
%

function h = figs_inialization(wp, map_h, map_w)
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

fig = figure(1);
axis([-W, W, -H, H]);
hold on
wp=wp';
h.map = plot(wp(1,:),wp(2,:), 'g')
hold on

xlabel('metres'), ylabel('metres')
set(fig, 'name', 'Hybrid Simulator')
set(fig,'units','points','position',[100,100,800,800])
h.obs= plot(0,-H,'b');%,'erasemode','xor'); % observations
end
