%{
    setting the moving objects
%}

% moving car #1 
lenght = 4;
width = 4;
j=1;
track_obj(j).type = 'car';
track_obj(j).x = [0.8*MAP_W; 0.2*MAP_H; pi/2];
track_obj(j).wp = WP';
track_obj(j).iwp = 1;
track_obj(j).H = setup_tracking_animations();
track_obj(j).G = 0;
track_obj(j).V = 10;
track_obj(j).LOOP = 4;
track_obj(j).size = [lenght, lenght, -lenght, -lenght; width/2, -width/2, -width/2,  width/2]; % vehicle animation


%
N_track_obj = length(track_obj);
tag_trk_obj = 1:N_track_obj;