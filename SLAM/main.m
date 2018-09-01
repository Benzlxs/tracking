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
xlabel('metres'), ylabel('metres')
set(fig, 'name', 'EKF-SLAM Simulator')
h= setup_animations();

veh= [4, 4 -4, -4; WHEELBASE/2, -WHEELBASE/2, -WHEELBASE/2,  WHEELBASE/2]; % vehicle animation

plines=[]; % for laser line animation
pcount=0;

% initialise states
%xtrue= zeros(3,1);
%x= zeros(3,1);
P= zeros(3);
xtrue = [0; -160; 0];
x = [0; -160; 0];

% tracking objects
x_track_1 = [];
track_1 = [0.8*MAP_W, 0;-0.8*MAP_W, 0; -0.8*MAP_W, -0.8*MAP_H; 0.8*MAP_W, -0.8*MAP_H; ]';

% initialise other variables and constants
dt= DT_CONTROLS; % change in time between predicts
dtsum= 0; % change in time since last observation
ftag= 1:size(lm,2); % identifier for each landmark
da_table= zeros(1,size(lm,2)); % data association table 
iwp= 1; % index to first waypoint 
G= 0; % initial steer angle
data= initialise_store(x,P,x); % stored data for off-line
QE= Q; RE= R; if SWITCH_INFLATE_NOISE, QE= 2*Q; RE= 8*R; end % inflate estimated noises (ie, add stabilising noise)
if SWITCH_SEED_RANDOM, randn('state',SWITCH_SEED_RANDOM), end

% main loop 
while iwp ~= 0
    
    % compute true data
    [G,iwp]= compute_steering(xtrue, wp, iwp, AT_WAYPOINT, G, RATEG, MAXG, dt);
    if iwp==0 & NUMBER_LOOPS > 1, iwp=1; NUMBER_LOOPS= NUMBER_LOOPS-1; end % perform loops: if final waypoint reached, go back to first
    xtrue= vehicle_model(xtrue, V,G, WHEELBASE,dt);
    [Vn,Gn]= add_control_noise(V,G,Q, SWITCH_CONTROL_NOISE);
    
    % EKF predict step
    [x,P]= predict (x,P, Vn,Gn,QE, WHEELBASE,dt);
    
    % if heading known, observe heading
    [x,P]= observe_heading(x,P, xtrue(3), SWITCH_HEADING_KNOWN);
    
    % EKF update step
    dtsum= dtsum + dt;
    if dtsum >= DT_OBSERVE
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
    end
    
    % offline data store
    data= store_data(data, x, P, xtrue);
    
    % plots
    xt= transformtoglobal(veh,xtrue);
    xv= transformtoglobal(veh,x(1:3));
    set(h.xt, 'xdata', xt(1,:), 'ydata', xt(2,:))
    set(h.xv, 'xdata', xv(1,:), 'ydata', xv(2,:))
    set(h.xf, 'xdata', x(4:2:end), 'ydata', x(5:2:end))
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
%

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
function data= initialise_store(x,P, xtrue)
% offline storage initialisation
data.i=1;
data.path= x;
data.true= xtrue;
data.state(1).x= x;
%data.state(1).P= P;
data.state(1).P= diag(P);
end
%
%
function h= setup_animations()
x_s = 0;
h.xt= patch(0,x_s,'b','erasemode','xor'); % vehicle true
h.xv= patch(0,x_s,'r','erasemode','xor'); % vehicle estimate
h.pth= plot(0,x_s,'k.','markersize',2,'erasemode','background'); % vehicle path estimate
h.obs= plot(0,x_s,'r','erasemode','xor'); % observations
h.xf= plot(0,0,'r+','erasemode','xor'); % estimated features
h.cov= plot(0,0,'r','erasemode','xor'); % covariance ellipses
end

%
%
function fig = figs_inialization(map_dir, map_h, map_w)
% load the road point and landmarks
global wp lm;
W = map_w, H = map_h; 
C = [0.5, 0.5, 0.5];  % color for map
load(map_dir);
assert(size(lm,2)>=10, 'The number of landmarks should be over 10');
assert(size(wp,2)>=4,'The number of way points shold be over 4'); 

fig = figure;
axis([-W, W, -H, H]);
% plotting the robot maps
lm = plot_mapping(W,H,C);
hold on
% plotting landmarks and way points
wp=[0,-0.8*H;  0.15*W, -0.8*H; 0.3*W, -0.4*H; 0.2*W, -0.15*H; -0.75*W, -0.15*H; -0.75*W, 0.75*H;
    -0.75*W, 0.77*H; -0.15*W, 0.77*H;  -0.35*W, 0.4*H; -0.15*W, 0.08*H;
     0.75*W, 0.08*H; 0.75*W, -0.15*H; -0.2*W, -0.15*H;  -0.35*W, - 0.4*H;
      -0.15*W, - 0.8*H; 0,-0.8*H ];
wp=wp'
plot(wp(1,:),wp(2,:), 'g', wp(1,:),wp(2,:),'g.')
hold on
plot(lm(1,:),lm(2,:),'b*')
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
function lm=plot_mapping(W,H,C)

% plotting the surroundings
x_1 = [];
y_1 = [];
num = 6;

wall_ratio = 0.15;
x1 = [-W, W, W, -W];
y1 = [ H, H, H - wall_ratio*H, H - wall_ratio*H];
patch(x1, y1, C);

x1 = [-W, W, W, -W];
y1 = [ -H, -H, -H + wall_ratio*H, -H + wall_ratio*H];
patch(x1, y1, C);

x1 = [-W, -W, -W+wall_ratio*W, -W+wall_ratio*W];
y1 = [ H,  -H, -H  ,  H ];
patch(x1, y1, C);

x1 = [W, W, W-wall_ratio*W, W-wall_ratio*W];
y1 = [ H,  -H, -H,  H ];
patch(x1, y1, C);

% plotting inland maps
x1 = [ 0.3*W, 0.6*W, 0.6*W, 0.3*W, 0.4*W];
y1 = [ 0.6*H, 0.6*H, 0.2*H, 0.2*H, 0.4*H];
patch(x1, y1, C);

% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end

 
x1 = [ 0.3*W, 0.6*W, 0.6*W, 0.3*W, 0.4*W];
y1 = -[ 0.6*H, 0.6*H, 0.2*H, 0.2*H, 0.4*H];
patch(x1, y1, C);

% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end


x1 = -[ 0.3*W, 0.6*W, 0.6*W, 0.3*W, 0.4*W];
y1 = [ 0.6*H, 0.6*H, 0.2*H, 0.2*H, 0.4*H];
patch(x1, y1, C);
% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end


x1 = -[ 0.3*W, 0.6*W, 0.6*W, 0.3*W, 0.4*W];
y1 = -[ 0.6*H, 0.6*H, 0.2*H, 0.2*H, 0.4*H];
patch(x1, y1, C);
len = length(x1);
% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end


% 
x1 = [ 0.25*W, 0.15*W, -0.15*W, -0.25*W, -0.15*W, 0.15*W];
y1 = [ 0.4*H, 0.6*H, 0.6*H, 0.4*H, 0.2*H, 0.2*H];
patch(x1, y1, C);
% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end


% 
x1 = [ 0.25*W, 0.15*W, -0.15*W, -0.25*W, -0.15*W, 0.15*W];
y1 = -[ 0.4*H, 0.6*H, 0.6*H, 0.4*H, 0.2*H, 0.2*H];
patch(x1, y1, C);
% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end


x1 = [0.6*W, 0.6*W, -0.6*W, -0.6*W];
y1 = [ 0.01*H, -0.01*H, -0.01*H, 0.01*H];
patch(x1, y1, C);
% generate landmarks
len = length(x1);
for i = 1:len
    if i==len
        x_inter = linspace(x1(i), x1(1), num);
        y_inter = linspace(y1(i), y1(1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    else
        x_inter = linspace(x1(i), x1(i+1), num);
        y_inter = linspace(y1(i), y1(i+1), num);
        x_inter(num) = [];
        y_inter(num) = [];
        x_1 = [x_1, x_inter];
        y_1 = [y_1, y_inter];
    end
end
lm = [x_1;y_1]
end