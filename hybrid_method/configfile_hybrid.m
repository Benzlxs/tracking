%{
 configfile for hybrid method
    parameters for map and moving object
%}

% Map paramters
MAP_H = 200;   % height of map
MAP_W = 200;   % width of map
up_lim = 0.6;
down_lim = 0.2;
WP = [0.8*MAP_W, down_lim*MAP_H; 0.55*MAP_W,up_lim*MAP_H; 0.25*MAP_W,down_lim*MAP_H; 0.0*MAP_W,up_lim*MAP_H; -0.25*MAP_W,down_lim*MAP_H; -0.55*MAP_W,up_lim*MAP_H;
     -0.8*MAP_W, down_lim*MAP_H; -0.8*MAP_W, -down_lim*MAP_H; -0.55*MAP_W, -up_lim*MAP_H; -0.25*MAP_W, -down_lim*MAP_H; 0.0*MAP_W, -up_lim*MAP_H; 0.25*MAP_W,-down_lim*MAP_H;
      0.55*MAP_W, -up_lim*MAP_H;  0.8*MAP_W, -down_lim*MAP_H;0.8*MAP_W, down_lim*MAP_H];
  
% control parameters
V= 5; % m/s
MAXG= 30*pi/180; % radians, maximum steering angle (-MAXG < g < MAXG)
RATEG= 20*pi/180; % rad/s, maximum rate of change in steer angle
WHEELBASE= 4; % metres, vehicle wheel-base
DT_CONTROLS= 0.025; % seconds, time interval between control signals  

% tracking noise
acc_noise= 1.2; % m^2/s
ang_velo= (50.0*pi/180); % radians
x_noise = 2; % x and y uncertainty
y_noise = 2;
Q_trk = [ x_noise^2,   0,     0,       0;
          0,   y_noise^2,     0,       0;
          0,    0, ang_velo^2,   0;
          0,    0,     0, acc_noise^2] ; 

% control noises
sigmaV= 0.3; % m/s
sigmaG= (3.0*pi/180); % radians
Q= [sigmaV^2 0; 0 sigmaG^2];

% observation parameters
MAX_RANGE= 50.0; % metres
DT_OBSERVE= 8*DT_CONTROLS; % seconds, time interval between observations

% observation noises
sigmaR= 0.1; % metres
sigmaB= (1.0*pi/180); % radians
R= [sigmaR^2 0; 0 sigmaB^2];

% data association innovation gates (Mahalanobis distances)
GATE_REJECT= 4.0; % maximum distance for association
GATE_AUGMENT= 25.0; % minimum distance for creation of new feature

% waypoint proximity
AT_WAYPOINT= 1.0; % metres, distance from current waypoint at which to switch to next waypoint
NUMBER_LOOPS= 2; % number of loops through the waypoint list

% hybrid model parameters
sigmaX = 0.05; % meter
sigmaY = 0.05; % meter
sigmaB = (1.0*pi/180); % radians
R_ed = [sigmaX^2 0 0; 0 sigmaY^2 0; 0 0 sigmaB^2];
sigmaX = 0.5; % meter
sigmaY = 0.5; % meter
R_cd = [sigmaX^2 0; 0 sigmaY^2];

R_b = (10.0*pi/180)^2; % bearing inializaiton uncertainty
R_v = (2.0)^2;  % velocity inialization uncertainty, there is no observation about it at beginning, so it should be large.

% Pattern of hybrid method
PATTEN = 0; % 0: ED for inialization, 1: CD for inialization
