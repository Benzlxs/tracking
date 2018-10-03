veh= [4, 4 -4, -4; WHEELBASE/2, -WHEELBASE/2, -WHEELBASE/2,  WHEELBASE/2]; % vehicle animation

plines=[]; % for laser line animation
pcount=0;

P= zeros(3);
xtrue = [0; -160; 0];
x = [0; -160; 0];

% tracking states and covariance should be empty
x_trk = [];  % will be n*m, n is the number of tracked objects, m is the number of state
P_trk = [];  % will be the n*m*m,  
% acceleration of velocity and angle rate
acc_noise= 1.2; % m^2/s
ang_velo= (50.0*pi/180); % radians
Q_trk = [ 1.0,   0,     0,    0;
          0,   1.0,     0,    0;
          0,   0, ang_velo^2, 0;
          0,   0,     0, acc_noise^2] ; 
% give big noise on process model so that trusting less on tracking object
% states, to use observed measurement to correct object quickly
      
% R_trk = [1.5,  0;
%          0 , 1.5] ;      % uncertainty about velocity and angle
R_trk = R ;
      
count_trk = [];   % n*1, to count how long the moving objects have not been observed

ind_trk_obj = [];  % n*1 to record the index of real objects

GATE_REJECT_TRK = 2; % maximum distance for association

GATE_AUGMENT_TRK = 4; % minimum distance for creation of new feature
% num to delete the object
num_del = 50;

% initialise other variables and constants
dt= DT_CONTROLS; % change in time between predicts
dtsum= 0; % change in time since last observation
ftag= 1:size(lm,2); % identifier for each landmark
da_table= zeros(1,size(lm,2)); % data association table 
mot_table = zeros(1, N_track_obj);
iwp= 1; % index to first waypoint 
G= 0; % initial steer angle
data= initialise_store(x,P,x); % stored data for off-line
QE= Q; RE= R; if SWITCH_INFLATE_NOISE, QE= 2*Q; RE= 8*R; end % inflate estimated noises (ie, add stabilising noise)
if SWITCH_SEED_RANDOM, randn('state',SWITCH_SEED_RANDOM), end


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