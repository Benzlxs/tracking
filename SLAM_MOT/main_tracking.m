%%
% This script is created originally on 29/08/2018; 
% version='1.0';
% 
% Author: Xuesong(Ben) Li, (benzlee08@gmail.com)
% University: UNSW
% All rights reserved

%Algorithm description: simulation of SLAM and object tracking
% 1. build map
% 2. set landmark and road point
% 3. run slam


% one branch one check


%% main body of algorithm
function []=main(map_dir)

clc, clear;
close all;
global wp lm;

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
n_obj_pre = 0;
% main loop 
while iwp ~= 0
    
    %% tracking objects 
    for j=1:N_track_obj
        [track_obj(j).G, track_obj(j).iwp]= compute_steering(track_obj(j).x, track_obj(j).wp, track_obj(j).iwp, AT_WAYPOINT, track_obj(j).G, RATEG, MAXG, dt);
        if track_obj(j).iwp==0 & track_obj(j).LOOP > 1, track_obj(j).iwp = 1; track_obj(j).LOOP = track_obj(j).LOOP ; end % perform loops: if final waypoint reached, go back to first
        track_obj(j).x= vehicle_model(track_obj(j).x,  track_obj(j).V , track_obj(j).G, WHEELBASE,dt);
        % plots
        xxtt = transformtoglobal(track_obj(j).size, track_obj(j).x);
        set(track_obj(j).H.xt_t1, 'xdata', xxtt(1,:), 'ydata', xxtt(2,:))
    end

    %% compute true data
    [G,iwp]= compute_steering(xtrue, wp, iwp, AT_WAYPOINT, G, RATEG, MAXG, dt);
    if iwp==0 & NUMBER_LOOPS > 1, iwp=1; NUMBER_LOOPS= NUMBER_LOOPS-1; end % perform loops: if final waypoint reached, go back to first
    xtrue= vehicle_model(xtrue, V,G, WHEELBASE,dt);
    [Vn,Gn]= add_control_noise(V,G,Q, SWITCH_CONTROL_NOISE);    
    
    
    %% EKF predict step
    [x,P]= predict (x,P, Vn,Gn,QE, WHEELBASE,dt);
    
    %% prediction of moving objects
    n_obj = size(x_trk,1);
    for i =1:n_obj
        %TO-DO: predict moving object one by one
        [x_temp,P_temp]= predict_moving_object (x_trk(i,:), squeeze(P_trk(i,:,:)), Q_trk, dt);  
        x_trk(i,:) = x_temp;
        P_trk(i,:,:) = P_temp; 
    end
    
    % if heading known, observe heading
    [x,P]= observe_heading(x,P, xtrue(3), SWITCH_HEADING_KNOWN);
    
    %% EKF update step
    dtsum= dtsum + dt;
    if dtsum >= DT_OBSERVE
        %% SLAM system
        dtsum= 0;
        [z,ftag_visible]= get_observations(xtrue, lm, ftag, MAX_RANGE);
        z= add_observation_noise(z,R, SWITCH_SENSOR_NOISE);
    
        if SWITCH_ASSOCIATION_KNOWN == 1
            [zf,idf,zn, da_table]= data_associate_known(x,z,ftag_visible, da_table);
        else
            [zf,idf, zn]= data_associate(x,P,z,RE, GATE_REJECT, GATE_AUGMENT); 
        end

        if SWITCH_USE_IEKF == 1
            [x,P]= update_iekf(x,P,zf,RE,idf, 5);
        else
            [x,P]= update(x,P,zf,RE,idf, SWITCH_BATCH_UPDATE); 
        end
        [x,P]= augment(x,P, zn,RE); 
        
        %% tracking system
        for k = 1:n_obj
            count_trk(k) = count_trk(k) + 1;
        end
        % getting the measurements from moving target
        [z_trk_obj, ind_trk_z] = get_observations_tracking_obj(xtrue, track_obj, tag_trk_obj, MAX_RANGE);
        z_trk_obj = add_observation_noise(z_trk_obj, R, SWITCH_SENSOR_NOISE); 

        % data association
        [zf_trk , idf_trk, zn_trk, zn_ind]= data_associate_tracking_obj(x(1:3), x_trk, P_trk, z_trk_obj, ind_trk_z, RE, GATE_REJECT_TRK, GATE_AUGMENT_TRK); 
        
        for k = 1:size(idf_trk,2)
            count_trk(idf_trk(k)) = 0 ; 
        end
        
        % update
        [x_trk , P_trk]= update_tracking_obj(x(1:3), x_trk , P_trk, zf_trk, RE, idf_trk);
        % augmentation and deletion
        [x_trk , P_trk, count_trk, ind_trk_obj]= augment_tracking_obj(x(1:3), P(1:3,1:3), x_trk , P_trk, zn_trk, R_trk, count_trk, zn_ind, ind_trk_obj); 
        [x_trk , P_trk, count_trk, ind_trk_obj] = del_tracking_obj(x_trk , P_trk, count_trk, num_del, ind_trk_obj);
    end
    
    %% plotting
    % offline data store
    data= store_data(data, x, P, xtrue);
    
    % plots
    xt= transformtoglobal(veh,xtrue);
    xv= transformtoglobal(veh,x(1:3));
    set(h.xt, 'xdata', xt(1,:), 'ydata', xt(2,:))
    set(h.xv, 'xdata', xv(1,:), 'ydata', xv(2,:))
    set(h.xf, 'xdata', x(4:2:end), 'ydata', x(5:2:end))
    
    % plotting tracked objects
    n_obj = size(x_trk,1);
    if n_obj~= n_obj_pre
        for i =1:n_obj_pre
            set( fig_hs(i).car,'xdata',0, 'ydata', 0);
            set( fig_hs(i).elliphse,'xdata', 0, 'ydata',0);
        end
    end
    
    for i =1:n_obj
        %TO-DO: predict moving object one by one
        
        ij = ind_trk_obj(i);
        x_obj=transformtoglobal(track_obj(ij).size, x_trk(i,:));
        %set(track_obj(ij).H.xv_t1,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        set( fig_hs(i).car,'xdata', x_obj(1,:), 'ydata', x_obj(2,:));
        p_conv= make_covariance_ellipses_tracking_obj(x_trk(i,:), squeeze(P_trk(i,:,:)));
        %set(track_obj(ij).H.cov_t1,'xdata', p_conv(1,:), 'ydata', p_conv(2,:));
        set( fig_hs(i).elliphse,'xdata', p_conv(1,:), 'ydata', p_conv(2,:));
        
        % vanish the deleted objects 
        %vanish_img(track_obj, ind_trk_obj, count_trk,num_del )
        if count_trk(i) >= num_del-4
            set( fig_hs(i).car,'xdata',0, 'ydata', 0);
            set( fig_hs(i).elliphse,'xdata', 0, 'ydata',0);
        end
    end
    n_obj_pre = n_obj;
    
    ptmp= make_covariance_ellipses(x(1:3),P(1:3,1:3));
    pcov(:,1:size(ptmp,2))= ptmp;
    if dtsum==0
        set(h.cov, 'xdata', pcov(1,:), 'ydata', pcov(2,:)) 
        pcount= pcount+1;
        if pcount == 15
            set(h.pth, 'xdata', data.path(1,1:data.i), 'ydata', data.path(2,1:data.i))    
            pcount=0;
        end
        if ~isempty(z)
            plines= make_laser_lines (z,x(1:3));
            if ~isempty(z_trk_obj)
                pp = make_laser_lines (z_trk_obj,x(1:3));
                plines = [plines pp];
            end
            set(h.obs, 'xdata', plines(1,:), 'ydata', plines(2,:))
            pcov= make_covariance_ellipses(x,P);
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

function p= make_covariance_ellipses(x,P)
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


