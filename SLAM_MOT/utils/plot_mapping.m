function lm=plot_mapping(W,H,C)

% plotting the surroundings
x_1 = [];
y_1 = [];
num = 6;
n_central = 14;
n_edge = 18;

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
num = n_central;
% central regions
len = length(x1);
for i = 2
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


% generate the surrounding landmarls
x1 = [-(1-wall_ratio)*W, (1-wall_ratio)*W, (1-wall_ratio)*W, -(1-wall_ratio)*W];
y1 = [ (1-wall_ratio)*H, (1-wall_ratio)*H, - (1-wall_ratio)*H, - (1-wall_ratio)*H];

num = n_edge;
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




lm = [x_1;y_1];
end