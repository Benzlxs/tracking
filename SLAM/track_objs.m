% tracking car1 
lenght = 4;
width = 4;
j=1;
track_obj(j).type = 'car';
track_obj(j).x = [0.7*MAP_W; -0.1*MAP_H; pi];
track_obj(j).wp = [0.7*MAP_W, -0.1*MAP_H;-0.7*MAP_W, -0.1*MAP_H; -0.7*MAP_W, -0.7*MAP_H; 0.7*MAP_W, -0.7*MAP_H; ]';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 3;
track_obj(j).LOOP = 4;
track_obj(j).size = [lenght, lenght, -lenght, -lenght; width/2, -width/2, -width/2,  width/2]; % vehicle animation
%track_obj(1).data=

% tracking car2 
lenght = 6;
width = 8;
j=2;
track_obj(j).type = 'car';
track_obj(j).x = [0.7*MAP_W; 0.7*MAP_H; pi];
track_obj(j).wp = [0.7*MAP_W, 0.7*MAP_H; -0.7*MAP_W, 0.7*MAP_H; -0.7*MAP_W, 0.1*MAP_H; 0.7*MAP_W, 0.1*MAP_H; ]';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 10;
track_obj(j).LOOP = 4;
track_obj(j).size = [lenght, lenght, -lenght, -lenght; width/2, -width/2, -width/2,  width/2]; % vehicle animation

% tracking car3
lenght = 5;
width = 7;
j=3;
track_obj(j).type = 'car';
track_obj(j).x = [0.7*MAP_W; 0.13*MAP_H; pi];
track_obj(j).wp = [0.7*MAP_W, 0.13*MAP_H; -0.7*MAP_W, 0.13*MAP_H; -0.7*MAP_W, -0.04*MAP_H; 0.7*MAP_W, -0.04*MAP_H; ]';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 8;
track_obj(j).LOOP = 6;
track_obj(j).size = [lenght, lenght, -lenght, -lenght; width/2, -width/2, -width/2,  width/2]; % vehicle animation



% track_pedestrain 1
edge=3;
j=4;
track_obj(j).type = 'pedestrain';
track_obj(j).x =  [ -0.1*MAP_W; 0.05*MAP_H; pi];
track_obj(j).wp = [ -0.1*MAP_W, 0.05*MAP_H; -0.4*MAP_W, 0.05*MAP_H; -0.4*MAP_W, 0.16*MAP_H; -0.1*MAP_W, 0.16*MAP_H; ]';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 2;
track_obj(j).LOOP = 4;
track_obj(j).size = [edge, -edge, -edge; 0, edge/2, -edge/2]; % vehicle animation

% tracking car3
lenght = 6;
width = 6;
j=5;
track_obj(j).type = 'car';
track_obj(j).x = [0; -0.85*MAP_H; 0];
track_obj(j).wp = [0,-0.85*MAP_H;  0.18*MAP_W, -0.85*MAP_H; 0.33*MAP_W, -0.4*MAP_H; 0.23*MAP_W, -0.18*MAP_H; -0.23*MAP_W, -0.15*MAP_H;  -0.38*MAP_W, - 0.4*MAP_H; -0.18*MAP_W, - 0.85*MAP_H ]';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 7;
track_obj(j).LOOP = 10;
track_obj(j).size = [lenght, lenght, -lenght, -lenght; width/2, -width/2, -width/2,  width/2]; % vehicle animation

 




%
N_track_obj = length(track_obj);
tag_trk_obj = 1:N_track_obj


% create a set of handle of plotting for tracked object
set_n = 20;
fig_hs=[];
for i =1:set_n
    fig_hs(i).car = patch(0,0,'r'); % tracked vehicle estimate
    fig_hs(i).elliphse= plot(0,0,'b'); % covariance ellipses
end


