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



% create a set of handle of plotting for tracked object
set_n = 20;
fig_hs=[];
for i =1:set_n
    fig_hs(i).car = patch(0,0,'r'); % tracked vehicle estimate
    fig_hs(i).elliphse= plot(0,0,'b'); % covariance ellipses
end

for i = N_track_obj
    h1 = figure(2*i)
    fig_dif(i).x = plot(0,0,'b');
    xlabel('count'), ylabel('X error')
    set(h1, 'name', sprintf('the error of object %d',i));
    h2 = figure(2*i+1)
    fig_dif(i).p = plot(0,0,'r');
    xlabel('count'), ylabel('P uncertainty')
    set(h2, 'name', sprintf('the uncertainty of object %d',i));        
end