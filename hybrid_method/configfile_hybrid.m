%{
 configfile for hybrid method
    parameters for map and moving object
%}

% Map paramters
MAP_H = 200;   % height of map
MAP_W = 200;   % width of map
up_lim = 0.5;
down_lim = 0.3;
WP = [0.8*MAP_W, down_lim*MAP_H; 0.75*MAP_W,up_lim*MAP_H; 0.65*MAP_W,down_lim*MAP_H;  0.55*MAP_W,up_lim*MAP_H; 
     0.45*MAP_W,down_lim*MAP_H; 0.35*MAP_W,up_lim*MAP_H; 0.25*MAP_W,down_lim*MAP_H; 0.0*MAP_W,up_lim*MAP_H;
     -0.25*MAP_W,down_lim*MAP_H; -0.35*MAP_W,up_lim*MAP_H; -0.45*MAP_W,down_lim*MAP_H; -0.55*MAP_W,up_lim*MAP_H;
     -0.65*MAP_W,down_lim*MAP_H; -0.75*MAP_W,up_lim*MAP_H; -0.8*MAP_W, down_lim*MAP_H; 
     -0.8*MAP_W, -down_lim*MAP_H; -0.55*MAP_W, -up_lim*MAP_H; -0.25*MAP_W, -down_lim*MAP_H; 0.0*MAP_W, -up_lim*MAP_H; 0.25*MAP_W,-down_lim*MAP_H;
     0.55*MAP_W, -up_lim*MAP_H;  0.8*MAP_W, -down_lim*MAP_H;0.8*MAP_W, down_lim*MAP_H];
% WP = [0.8*MAP_W, down_lim*MAP_H;  0.55*MAP_W,up_lim*MAP_H; 0.25*MAP_W,down_lim*MAP_H; 0.0*MAP_W,up_lim*MAP_H; -0.25*MAP_W,down_lim*MAP_H; -0.55*MAP_W,up_lim*MAP_H;
%       -0.8*MAP_W, down_lim*MAP_H; -0.8*MAP_W, -down_lim*MAP_H; -0.55*MAP_W, -up_lim*MAP_H; -0.25*MAP_W, -down_lim*MAP_H; 0.0*MAP_W, -up_lim*MAP_H; 0.25*MAP_W,-down_lim*MAP_H;
%        0.55*MAP_W, -up_lim*MAP_H;  0.8*MAP_W, -down_lim*MAP_H;0.8*MAP_W, down_lim*MAP_H]';   
% T = 50;
% WP = (2*rand(2,T) - 1) * MAP_H;
% start = [0.8*MAP_W; down_lim*MAP_H];
% WP = [start  WP];

% control parameters
V= 5; % m/s
MAXG= 30*pi/180; % radians, maximum steering angle (-MAXG < g < MAXG)
RATEG= 20*pi/180; % rad/s, maximum rate of change in steer angle
WHEELBASE= 4; % metres, vehicle wheel-base
DT_CONTROLS= 0.025; % seconds, time interval between control signals  

% tracking noise
acc_noise= 1.2; % m^2/s
ang_velo= (50.0*pi/180); % radians
x_noise = 1; % x and y uncertainty
y_noise = 1;
Q_trk = [ x_noise^2,   0,     0,       0;
          0,   y_noise^2,     0,       0;
          0,    0, ang_velo^2,   0;
          0,    0,     0, acc_noise^2] ; 

Q_trk_ed = Q_trk;
Q_trk_cd = 2*Q_trk;

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
AT_WAYPOINT= 2.0; % metres, distance from current waypoint at which to switch to next waypoint
NUMBER_LOOPS= 2; % number of loops through the waypoint list

% hybrid model parameters
sigmaX = 0.08; % meter
sigmaY = 0.08; % meter
sigmaB = (1.0*pi/180); % radians
R_ed = [sigmaX^2 0 0; 0 sigmaY^2 0; 0 0 sigmaB^2];
sigmaX = 0.2; % meter
sigmaY = 0.2; % meter
R_cd =  40*[sigmaX^2 0; 0 sigmaY^2];

R_b = (10.0*pi/180)^2; % bearing inializaiton uncertainty
R_v = (2.0)^2;  % velocity inialization uncertainty, there is no observation about it at beginning, so it should be large


% uncertainty for switch
p_2_ed = 0.9; %0.34
p_2_cd = 0.1;


