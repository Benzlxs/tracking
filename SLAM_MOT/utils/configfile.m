%%% Configuration file
%%% Permits various adjustments to parameters of the SLAM algorithm.
%%% See ekfslam_sim.m for more information

% control parameters
V= 5; % m/s
MAXG= 30*pi/180; % radians, maximum steering angle (-MAXG < g < MAXG)
RATEG= 20*pi/180; % rad/s, maximum rate of change in steer angle
WHEELBASE= 4; % metres, vehicle wheel-base
DT_CONTROLS= 0.025; % seconds, time interval between control signals

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
% For 2-D observation:
%   - common gates are: 1-sigma (1.0), 2-sigma (4.0), 3-sigma (9.0), 4-sigma (16.0)
%   - percent probability mass is: 1-sigma bounds 40%, 2-sigma 86%, 3-sigma 99%, 4-sigma 99.9%.

% waypoint proximity
AT_WAYPOINT= 1.0; % metres, distance from current waypoint at which to switch to next waypoint
NUMBER_LOOPS= 2; % number of loops through the waypoint list

% switches
SWITCH_CONTROL_NOISE= 1; % if 0, velocity and gamma are perfect
SWITCH_SENSOR_NOISE = 1; % if 0, measurements are perfect
SWITCH_INFLATE_NOISE= 0; % if 1, the estimated Q and R are inflated (ie, add stabilising noise)
SWITCH_HEADING_KNOWN= 0; % if 1, the vehicle heading is observed directly at each iteration
SWITCH_ASSOCIATION_KNOWN= 1; % if 1, associations are given, if 0, they are estimated using gates
SWITCH_BATCH_UPDATE= 1; % if 1, process scan in batch, if 0, process sequentially
SWITCH_SEED_RANDOM= 0; % if not 0, seed the randn() with its value at beginning of simulation (for repeatability)
SWITCH_USE_IEKF= 0; % if 1, use iterated EKF for updates, if 0, use normal EKF


MAP_H = 200;   % height of map
MAP_W = 200;   % width of map
