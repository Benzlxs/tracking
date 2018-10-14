%%
% This script is created originally on 29/09/2018; 
% version='1.0';
% 
% Author: Xuesong(Ben) Li, (benzlee08@gmail.com)
% University: UNSW
% All rights reserved

%Algorithm description: SLAM and object tracking together, full KF is
%performed
% 1. build map
% 2. set landmark and road point
% 3. run slam


% one branch one check


%% main body of algorithm
function []=main(map_dir)

clc, clear;
close all;
global wp lm;

rng(0,'twister')
if nargin==0
    map_dir='./map.mat';
end
%addpath('../openslam_bailey-slam/ekfslam_v1/')
addpath('./utils')

format compact
configfile;  %% the file is used to configure the ekf-slam, make sure importing successfully

fig = figs_inialization(map_dir, MAP_H, MAP_W);


h= setup_animations();

% set tracked objects
track_objs;

% initialise states
initialization_params;

% Q_trk = [ 2,   0,     0,    0;
%           0,   2,     0,    0;
%           0,   0,     1,    0;
%           0,   0,     0,    1] ; 

% Q_trk = 4*Q_trk;
% 
% R_trk =  [sigmaR^2 0; 0 sigmaB^2];
      
n_obj_pre = 0;

num_lm = 0;  % keep number of landmark
% main loop 
debug=0;
while iwp ~= 0    
    %% tracking objects 
    for j=1:N_track_obj
        [track_obj(j).G, track_obj(j).iwp]= compute_steering(track_obj(j).x, track_obj(j).wp, track_obj(j).iwp, AT_WAYPOINT, track_obj(j).G, RATEG, MAXG, dt);
        if track_obj(j).iwp==0 & track_obj(j).LOOP > 1, track_obj(j).iwp = 1; track_obj(j).LOOP = track_obj(j).LOOP ; end % perform loops: if final waypoint reached, go back to first
        track_obj(j).x= vehicle_model(track_obj(j).x,  track_obj(j).V, track_obj(j).G, WHEELBASE, dt);
        % plots
        xxtt = transformtoglobal(track_obj(j).size, track_obj(j).x);
        set(track_obj(j).H.xt_t1, 'xdata', xxtt(1,:), 'ydata', xxtt(2,:))
    end

    %% compute true data
    [G,iwp]= compute_steering(xtrue, wp, iwp, AT_WAYPOINT, G, RATEG, MAXG, dt);
    if iwp==0 & NUMBER_LOOPS > 1, iwp=1; NUMBER_LOOPS= NUMBER_LOOPS-1; end % perform loops: if final waypoint reached, go back to first
    xtrue= vehicle_model(xtrue, V,G, WHEELBASE,dt);
    [Vn,Gn]= add_control_noise(V,G,Q, SWITCH_CONTROL_NOISE);    
        
    %% EKF predict step of SLAM and moving object tracking
    [x,P]= predict_slam_mot( x, P, Vn, Gn, QE, WHEELBASE, dt, num_lm, Q_trk);

    %% EKF update step
    dtsum= dtsum + dt;
    if dtsum >= DT_OBSERVE
        %% Get simulated Observation
        debug = debug + 1    
        dtsum= 0;
        % slam observation
        [z_lm ,ftag_visible, z_mot_obj, ind_mot_z]= get_observations_slam_mot(xtrue, lm, ftag, MAX_RANGE, num_lm, track_obj, tag_trk_obj);
        z_lm= add_observation_noise(z_lm,R, SWITCH_SENSOR_NOISE);
        z_mot_obj = add_observation_noise(z_mot_obj, R, SWITCH_SENSOR_NOISE);
        % tracking observation
        num_mot = (size(x,1) - 3 - num_lm*2)/4;
        for k = 1:num_mot
            count_trk(k) = count_trk(k) + 1;
        end
        %[z_mot_obj, ind_mot_z] = get_observations_tracking_obj(xtrue, track_obj, tag_trk_obj, MAX_RANGE);
                 
        %% Data association
        % slam data association  
        if SWITCH_ASSOCIATION_KNOWN == 1
            [zf,idf,zn, da_table]= data_associate_known(x(1:(3+2*num_lm)),z_lm,ftag_visible, da_table);
            [zf_mot , idf_mot, zn_mot, zn_ind, mot_table]= data_associate_slam_mot_known(x, z_mot_obj, ind_mot_z, num_lm, mot_table); 
        else        
            [zf,idf, zn]= data_associate_slam_mot_lm(x,P,z_lm,RE, GATE_REJECT, GATE_AUGMENT, num_lm); 
            [zf_mot , idf_mot, zn_mot, zn_ind]= data_associate_slam_mot_mot(x, P, z_mot_obj, ind_mot_z, RE, GATE_REJECT_TRK, GATE_AUGMENT_TRK, num_lm); 
        end       
        %% updating with full Kalman filtering
        if SWITCH_USE_IEKF == 1
            [x,P]= update_iekf_slam_mot(x,P,zf,RE,idf,zf_mot, RE, idf_mot, num_lm, 5);
        else
            [x,P]= update_slam_mot(x,P,zf, RE, idf, zf_mot, R_trk, idf_mot, SWITCH_BATCH_UPDATE, num_lm);        
        end
        
        % [x_trk , P_trk]= update_tracking_obj(x(1:3), x_trk , P_trk, zf_mot, RE, idf_mot);        
        %% augmentation and deleting
        [x,P]= augment_slam_mot_lm(x,P, zn,RE, num_lm); 
        num_lm = num_lm + size(zn,2);  % updating the number of landmark, after agumentation, the num_lm can be updated.
        for k = 1:size(idf_mot,2)
            count_trk(idf_mot(k)) = 0 ; 
        end

        % augmentation and deletion
        [x, P, count_trk, ind_trk_obj] = augment_slam_mot_mot(x, P, zn_mot, zn_ind, R_trk, num_lm, ind_trk_obj, count_trk); 
        if SWITCH_ASSOCIATION_KNOWN == 1
            [x, P, count_trk, ind_trk_obj, mot_table] = del_slam_mot_known(x, P, count_trk, num_del, ind_trk_obj, num_lm, mot_table);
        else
            [x, P, count_trk, ind_trk_obj] = del_slam_mot(x, P, count_trk, num_del, ind_trk_obj, num_lm);
        end
        
        
%       if mod(debug,1)==0
%             eigens = eigs(P)
%       end
    end
  
    %% plotting
    % offline data store
    data= store_data(data, x, P, xtrue);
    
    % plots
    xt= transformtoglobal(veh,xtrue);
    xv= transformtoglobal(veh,x(1:3));
    set(h.xt, 'xdata', xt(1,:), 'ydata', xt(2,:))
    set(h.xv, 'xdata', xv(1,:), 'ydata', xv(2,:))
    set(h.xf, 'xdata', x(4:2:(3+2*num_lm)), 'ydata', x(5:2:(num_lm)))
    
    % plotting tracked objects
    num_mot = (size(x,1) - 3 - num_lm*2)/4;
    if num_mot~= n_obj_pre
        for i =1:n_obj_pre
            set( fig_hs(i).car,'xdata',0, 'ydata', 0);
            set( fig_hs(i).elliphse,'xdata', 0, 'ydata',0);
        end
    end
    
    for i =1:num_mot
        %TO-DO: predict moving object one by one
        ind = 3 + 2*num_lm + 4*(i-1) + 1; 
        ij = ind_trk_obj(i);
        x_obj=transformtoglobal(track_obj(ij).size, x(ind:ind+3));
        %set(track_obj(ij).H.xv_t1,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        set( fig_hs(i).car,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        p_conv= make_covariance_ellipses_tracking_obj(x(ind:ind+3), P(ind:ind+3,ind:ind+3));
        %set(track_obj(ij).H.cov_t1,'xdata', p_conv(1,:), 'ydata', p_conv(2,:));
        set( fig_hs(i).elliphse,'xdata',p_conv(1,:), 'ydata', p_conv(2,:));
        
        % vanish the deleted objects 
        %vanish_img(track_obj, ind_trk_obj, count_trk,num_del )
        if count_trk(i) >= num_del-4
            set( fig_hs(i).car,'xdata',0, 'ydata', 0);
            set( fig_hs(i).elliphse,'xdata', 0, 'ydata',0);
        end
    end
    n_obj_pre = num_mot;
    
    ptmp= make_covariance_ellipses(x(1:3),P(1:3,1:3));
    pcov(:,1:size(ptmp,2))= ptmp;
    if dtsum==0
        set(h.cov, 'xdata', pcov(1,:), 'ydata', pcov(2,:)) ;
        pcount= pcount+1;
        if pcount == 15
            set(h.pth, 'xdata', data.path(1,1:data.i), 'ydata', data.path(2,1:data.i))    
            pcount=0;
        end
        if ~isempty(z_lm)
            plines= make_laser_lines (z_lm,x(1:3));
            if ~isempty(z_mot_obj)
                pp = make_laser_lines (z_mot_obj,x(1:3));
                plines = [plines pp];
            end
            set(h.obs, 'xdata', plines(1,:), 'ydata', plines(2,:))
            pcov= make_covariance_ellipses(x(1:(3+2*num_lm)), P(1:(3+2*num_lm),1:(3+2*num_lm)));
        end
    end
    drawnow
end

data= finalise_data(data);
set(h.pth, 'xdata', data.path(1,:), 'ydata', data.path(2,:))    


end
%
%% 
function []= vanish_img(track_obj, ind_trk_obj, count_trk,num_del)
% disvisable the remained image

num_obj = size(ind_trk_obj,1);
for i = 1:num_obj
    if count_trk(i) >= num_del-3
        ij = ind_trk_obj(i);
        set(track_obj(ij).H.xv_t1, 'xdata', 0, 'ydata', 0);
        set(track_obj(ij).H.cov_t1,'xdata', 0, 'ydata', 0);
    end
end

end


function p= make_laser_lines (rb,xv)
% compute set of line segments for laser range-bearing measurements
if isempty(rb), p=[]; return, end
len= size(rb,2);
lnes(1,:)= zeros(1,len)+ xv(1);
lnes(2,:)= zeros(1,len)+ xv(2);
lnes(3:4,:)= transformtoglobal([rb(1,:).*cos(rb(2,:)); rb(1,:).*sin(rb(2,:))], xv);
p= line_plot_conversion (lnes);
end
%
%

function p= make_covariance_ellipses(x, P)
% compute ellipses for plotting state covariances
N= 10;
inc= 2*pi/N;
phi= 0:inc:2*pi;

lenx= length(x);
lenf= (lenx-3)/2;
p= zeros (2,(lenf+1)*(N+2));

ii=1:N+2;
p(:,ii)= make_ellipse(x(1:2), P(1:2,1:2), 2, phi);

ctr= N+3;
for i=1:lenf
    ii= ctr:(ctr+N+1);
    jj= 2+2*i; jj= jj:jj+1;
    
    p(:,ii)= make_ellipse(x(jj), P(jj,jj), 2, phi);
    ctr= ctr+N+2;
end
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
function data= finalise_data(data)
% offline storage finalisation
data.path= data.path(:,1:data.i);
data.true= data.true(:,1:data.i);
end

%
%
function data= store_data(data, x, P, xtrue)
% add current data to offline storage
CHUNK= 5000;
if data.i == size(data.path,2) % grow array in chunks to amortise reallocation
    data.path= [data.path zeros(3,CHUNK)];
    data.true= [data.true zeros(3,CHUNK)];
end
i= data.i + 1;
data.i= i;
data.path(:,i)= x(1:3);
data.true(:,i)= xtrue;
data.state(i).x= x;
%data.state(i).P= P;
data.state(i).P= diag(P);
end
%
%



function h= setup_animations()
x_s = 0;
h.xt= patch(0,x_s,'b'); % vehicle true
h.xv= patch(0,x_s,'r'); % vehicle estimate
h.pth= plot(0,x_s,'k.','markersize',2);% ,'erasemode','background'); % vehicle path estimate
h.obs= plot(0,x_s,'r');%,'erasemode','xor'); % observations
h.xf= plot(0,0,'r+');%,'erasemode','xor'); % estimated features
h.cov= plot(0,0,'r');%,'erasemode','xor'); % covariance ellipses
end

%
%
function fig = figs_inialization(map_dir, map_h, map_w)
% load the road point and landmarks
global wp lm;
W = map_w;
H = map_h; 
C = [0.5, 0.5, 0.5];  % color for map
load(map_dir);
assert(size(lm,2)>=10, 'The number of landmarks should be over 10');
assert(size(wp,2)>=4,'The number of way points shold be over 4'); 

fig = figure;
%set(gcf, 'PaperSize', [4 2]);

axis([-W, W, -H, H]);
% plotting the robot maps
lm = plot_mapping(W,H,C);
hold on
% plotting landmarks and way points
wp=[0,-0.8*H;  0.15*W, -0.8*H; 0.3*W, -0.4*H; 0.2*W, -0.15*H; -0.75*W, -0.15*H; -0.75*W, 0.75*H;
    -0.75*W, 0.77*H; -0.15*W, 0.77*H;  -0.35*W, 0.4*H; -0.15*W, 0.08*H;
     0.75*W, 0.08*H; 0.75*W, -0.15*H; -0.2*W, -0.15*H;  -0.35*W, - 0.4*H;
      -0.15*W, - 0.8*H; 0,-0.8*H ];
wp=wp';
plot(wp(1,:),wp(2,:), 'g', wp(1,:),wp(2,:),'g.')
hold on
plot(lm(1,:),lm(2,:),'b*')

xlabel('metres'), ylabel('metres')
set(fig, 'name', 'EKF-SLAM Simulator')
set(fig,'units','points','position',[100,100,800,800])

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


