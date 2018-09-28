% This script is created originally on 14/06/2018
%{
Author: Xuesong LI <benzlee08@gmail.com>

Copyright (c) 2018 University of New South Wales, all rights reserved

The MIT License (MIT)

Copyright (c) 2018 Xuesong LI.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
%}

%{
---------------- Code Description --------------------
THe code is to create simulation platform for hybrid manner.
Simulation context includes background with consistent color, and two or
three moving object,'cropped stops', the moving direction and velocity are
randomly generated.
Low computation: using threshold or color to segment object quickly;
High computation: RCNN to detection object reliable.
The trajectory of stop sign movement is predefined. To track every object
just uisng simple segmenation, running EKF.
Then adding RCNN into tracking framework.
%}

% To-do-list:
% 1. Randomly generating several starting points in given blank image;
% 2. pre-defining the motion trajectory of every stop sign, and moving stop
% signs;
% 3. process model for states are constant velocity eastimated from last two
% movement;
% 4. using the simple segmenation alogrithm, based on red color channel, to
% segment stop signs out as the measurement, which means that measurement
% model is just simplest linear model;
% 5. runing the RCNN to classify object into different category which
% represents different process model.
% 
% 
% States: X = {x1, y1, c11, c12, x2, y2, c21, c22}
% the high-frequency segment updates x1, y1, x2 and y2
% the low-frequency RCNN update c11, c12, c21, c22.

function [] = main()

    clear, clc;

    % initlization the car position
    Cars_X0 = [200, 200 ;  1000, 1000];
    num_cars = 2;
    X0 = [ Cars_X0(1,1); Cars_X0(1,2); 0 ; 0; Cars_X0(2,1); Cars_X0(2,2); 0; 0];
    Width = 1300; Height = 1300;
    dir_cars = ['./data/car_black.jpg';'./data/car_white.jpg'];

    % initializaiton the plotting
    fighandles = figure_init(Width, Height);
    Cars = cars_init( Cars_X0, dir_cars);

    % initialization the kalman filter, it's linear process and observation
    % model, so Kalman filter is enough
    P = zeros(8,8);
    P(3,3) = 0.1;
    P(7,7) = 0.1;
    Q = diag([(0.1)^2, (0.1)^2, 0, 0, (0.1)^2, (0.1)^2, 0, 0]);  %% we assumen the process model doesn't influence objects' classification
    R_seg = diag([(0.1)^2, (0.1)^2, (0.1)^2, (0.1)^2]);
    R_nn  = diag([(0.1)^2, (0.1)^2]);

    % dt is 1 millisecond
    dt = 1;
    T=100000;
    X_gt=[];
    X_gt(:,1) = X0;
    X_gt_ctr(:,:,1) = Cars_X0;
    v_gt_p = 2*(rand(2,2)-0.5);

    X_estimate=[];
    X_estimate(:,1) = X0;
    X_estimate_ctr(:,:,1) = Cars_X0;
    v_estimate_p = zeros(2,2);

    % some tempor variable to debug code
    X_temp = [];
    X_temp(:,:,1) = Cars_X0;

    for i=2:T
        % simulation platform output the controlling signals, ground_truth
        %% Simulated process model
        v_gt = simulate_drive(v_gt_p,X_gt(:,i-1), Width, Height, num_cars);
        v_gt_p = v_gt;
        d_v_gt = [v_gt(1,1)*dt; v_gt(1,2)*dt; 0; 0; v_gt(2,1)*dt; v_gt(2,2)*dt; 0; 0];
        X_gt(:,i) = X_gt(:,i-1) + d_v_gt;
        X_gt_ctr(:,:,i) = [X_gt(1,i), X_gt(2,i); X_gt(5,i) X_gt(6,i)];

        %% assumed process model in real application
        d_v_estimate = [v_estimate_p(1,1)*dt; v_estimate_p(1,2)*dt; 0; 0; v_estimate_p(2,1)*dt; v_estimate_p(2,2)*dt; 0; 0];
        X_estimate(:,i) = X_estimate(:,i-1) + d_v_estimate;
        J = diag(ones(1,8));
        P = J*P*J' + Q;
        %X_estimate_ctr(:,:,i) = [X_estimate(1,i), X_estimate(2,i); X_estimate(5,i) X_estimate(6,i)];
        % getting the image
        img = get_process_image(Width, Height, X_gt_ctr(:,:,i), Cars);  % collecting usefull information from the real state

        % using the segmentation observation to update state at high frequency
        %% Using the segmenation to update the location
        seg_ctr = segmentation_img(img, num_cars);
        X_temp(:,:,i)= seg_ctr;
        % doing data association to find matcing objects, very simple
        dist_1 = (seg_ctr(1,1) - X_estimate(1,i))^2 + (seg_ctr(1,2) - X_estimate(2,i))^2 ;
        dist_2 = (seg_ctr(2,1) - X_estimate(1,i))^2 + (seg_ctr(2,2) - X_estimate(2,i))^2 ;
        if dist_1 < dist_2
            measure_ctr= [seg_ctr(1,1); seg_ctr(1,2); seg_ctr(2,1); seg_ctr(2,2)];
        else
            measure_ctr= [seg_ctr(2,1); seg_ctr(2,2); seg_ctr(1,1); seg_ctr(1,2)];
        end

        %measure_ctr= [seg_ctr(1,:)';seg_ctr(2,:)'];
        H = [[ 1, 0, 0, 0, 0, 0, 0, 0];
             [ 0, 1, 0, 0, 0, 0, 0, 0];
             [ 0, 0, 0, 0, 1, 0, 0, 0];
             [ 0, 0, 0, 0, 0, 1, 0, 0]
             ];
        z = measure_ctr - H*X_estimate(:,i);

        S = R_seg + H*P*H' ;
        iS= inv(S) ;              % iS = inv(S) ;   % in this case S is 1x1 so inv(S) is just 1/S
        K = P*H'*iS ;           % Kalman gain

        X_estimate(:,i) = X_estimate(:,i) + K*z;
        P = P-P*H'*iS*H*P ;     % update the Covariance

        X_estimate_ctr(:,:,i) = [X_estimate(1,i), X_estimate(2,i); X_estimate(5,i) X_estimate(6,i)];
        v_estimate_p = [X_estimate(1,i) - X_estimate(1,i-1), X_estimate(2,i) - X_estimate(2,i-1);
                        X_estimate(5,i) - X_estimate(5,i-1), X_estimate(6,i) - X_estimate(6,i-1) ];

        plot_car(fighandles, X_gt_ctr(:,:,i), Cars);
        plot_trajectory(fighandles, X_gt_ctr, X_estimate_ctr);
        plot_text(fighandles, X_estimate(:,i), num_cars);

        if mod(i,40)==0
            [class_center, class_confid] = NN_detection(img);
            % doing data association to find matcing objects, very simple
            dist_1 = (class_center(1,1) - X_estimate(1,i))^2 + (class_center(1,2) - X_estimate(2,i))^2 ;
            dist_2 = (class_center(2,1) - X_estimate(1,i))^2 + (class_center(2,2) - X_estimate(2,i))^2 ;
            if dist_1 < dist_2
                measure_confid= class_confid;
            else
                measure_confid= flipud(class_config);
            end
            H = [[ 0, 0, 1, 0, 0, 0, 0, 0];
                 [ 0, 0, 0, 0, 0, 0, 1, 0]
                 ];
            z = measure_confid - H*X_estimate(:,i);

            S = R_nn + H*P*H' ;
            iS= inv(S) ;              % iS = inv(S) ;   % in this case S is 1x1 so inv(S) is just 1/S
            K = P*H'*iS ;           % Kalman gain

            X_estimate(:,i) = X_estimate(:,i) + K*z;
            P = P-P*H'*iS*H*P ;     % update the Covariance          
        end

        %img = get_process_image(Width, Height, X_ctr(:,:,i), Cars);
        pause(0.01)

    end
    %img_nn = getting_img(Cars)
end


function fighandles = figure_init(Width, Height)
    % figure initilization
    figure(1); clf(); hold on;
    axis([0, Width, 0, Height]);
    fighandles.cars(1) = image('CData',[0 0 0],'XData',[200 201], 'YData',[200 201]);
    fighandles.cars(2) = image('CData',[0 0 0],'XData',[1 1], 'YData',[1 1]);
    fighandles.real_path= plot(0,0,'.r');
    fighandles.estimate_path= plot(0,0,'.b');
    
    fighandles.confid(1) = text(0,0,' ','Color','blue');
    fighandles.confid(2) = text(0,0,' ','Color', 'blue');
    
    set(gca, 'YDir','reverse')
end

function Cars = cars_init(Cars_X0, dir_cars)
    n_cars = size(dir_cars,1);
    Cars=[];
    for i = 1:n_cars
        img = imread(dir_cars(i,:));
        h = size(img,1);
        w = size(img,2);
        Cars(i).img = img;
        Cars(i).h = h;
        Cars(i).w = w;
    end
end

function plot_car(fighandles, X_ctr, Cars)
    n_cars = length(Cars);
    for i =1:n_cars
        w_img = Cars(i).w;
        h_img = Cars(i).h;
        w_start = X_ctr(i,1) - ceil(w_img/2);
        h_start = X_ctr(i,2) - ceil(h_img/2);
        set(fighandles.cars(i), 'CData',Cars(i).img,'XData',[w_start  w_start + w_img ], 'YData', [h_start  h_start + h_img]);
    end
   %fighandles.img = image('CData',black_car, 'XData', [bc_w_start bc_w_start+bc_w-1], 'YData', [bc_h_start bc_h_start+bc_h-1]);
    %set(fighandles.black_car,'CData',black_car, 'XData', [bc_w_start bc_w_start+bc_w-1], 'YData', [bc_h_start bc_h_start+bc_h-1])
end

function plot_trajectory(fighandles, X_gt_ctr, X_estimate_ctr)
    xx = reshape(X_gt_ctr(:,1,:), [], 1);
    yy = reshape(X_gt_ctr(:,2,:), [], 1);
    set(fighandles.real_path, 'xdata', xx,'ydata', yy);

    xx = reshape(X_estimate_ctr(:,1,:), [], 1);
    yy = reshape(X_estimate_ctr(:,2,:), [], 1);
    set(fighandles.estimate_path, 'xdata', xx,'ydata', yy);    
    
end

function plot_text(fighandles, X_estimate, num_cars)
    for i = 1:num_cars
        idx = (i - 1)*4;
        annotation = sprintf('car: (Confidence = %f)', X_estimate(idx+3));
        set(fighandles.confid(i),'Position',[X_estimate(idx+1)-150, X_estimate(idx+2) - 150],'String',annotation);
        
    end
end

% simulate the moving platform to generate the controlling signals
function v = simulate_drive(v_p,X, Width, Height, num_cars)
    %v = ones(2,2); 
    % randomly generate the controlling signals
    v = v_p;
    % to avoid collision
    dist = sqrt( (X(1)-X(5))^2 + (X(2)-X(6))^2);
    if dist < 400  %% going to collide 
         if X(1) > X(5)
             v(:,1) = [2,-2]*rand;
         else
             v(:,1) = [-2,2]*rand;
         end
         if X(2) > X(6)
             v(:,2) = [2, -2]*rand;
         else
             v(:,2) = [-2, 2]*rand;
         end
             
    end
    
    % removing out-of-boundary points
    for j = 1:num_cars
        idx = (j-1)*4+1;
        if  X(idx)<200
            v(j,1) = 4*rand;
        elseif X(idx) > Width-200
             v(j,1) = -4*rand;
        end
        idx = idx + 1;
        if  X(idx)<200
            v(j,2) = 4*rand;
        elseif X(idx) > Height-200
            v(j,2) = -4*rand;
        end
    end

end

function img=get_process_image(Width, Height, X_ctr,Cars)
    img = 128 * ones(Height, Width, 3, 'uint8');

    n_car = length(Cars);

    for i=1:n_car
        w_img = Cars(i).w;
        h_img = Cars(i).h;    
        wc_w_start = int32(X_ctr(i,1) - ceil(w_img/2));
        wc_h_start = int32(X_ctr(i,2) - ceil(h_img/2));

        img(wc_h_start:wc_h_start+h_img-1, wc_w_start:wc_w_start+w_img-1, :) = Cars(i).img;
    end

end

function seg_ctrl = segmentation_img(img, num_cars)
    I = rgb2gray(img);
    bw = edge(I,'canny');
    [yy xx] = find(bw==1);
    area = [xx, yy];
    [idx seg_ctrl] = kmeans(area, num_cars);
    %seg_ctrl = int32(seg_ctrl);

end
function [class_center, class_confid] = NN_detection(img)
    % loading the model
    model = load('fasterRCNNVehicleTrainingData.mat');
    % Read test image
    tic
    % Detect stop signs
    [bboxes,score, label] = detect(model.detector,img);
    toc
    % bboxes = [x, y, width, height], (x,y) is the upper-left corner
    class_confid = score;
    class_center = [bboxes(1,1)+bboxes(1,3)/2, bboxes(1,2)+bboxes(1,4)/2 ;
                    bboxes(2,1)+bboxes(2,3)/2, bboxes(2,2)+bboxes(2,4)/2];
    %class_center = [bboxes() ; ]
    % Display the detection results
%     [score, idx] = max(score);
%     %bbox = bboxes(idx, :);
%     annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
%     outputImage = insertObjectAnnotation(img, 'rectangle', bboxes, annotation,'LineWidth', 4, 'FontSize',27);
%     figure(3)
%     imshow(outputImage);
end