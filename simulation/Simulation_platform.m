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

clear, clc;

Width = 1300, Height = 1300;
image = 128 * ones(Width, Height, 3, 'uint8');
yellow_stop = imread('./data/s_yellow.jpg');
red_stop = imread('./data/s_red.jpg');

[y_h, y_w, c] = size(yellow_stop);
y_h_start = 160;
y_w_start = 220;
image(y_h_start:y_h_start+y_h-1, y_w_start:y_w_start+y_w-1, :) = yellow_stop;

red_stop = imresize(red_stop, [y_h, y_w]);
[r_h, r_w, c] = size(red_stop);
r_h_start = 360;
r_w_start = 320;
image(r_h_start:r_h_start+r_h-1, r_w_start:r_w_start+r_w-1, :) = red_stop;


% loading the model

load('rcnnStopSigns.mat','rcnn');   
% Read test image

tic
% Detect stop signs
[bboxes,score, label] = detect(rcnn,image);

toc


% Display the detection results
[score, idx] = max(score);

%bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(image, 'rectangle', bboxes(1:2,:), annotation, 'LineWidth', 4, 'FontSize',27 );

figure
imshow(outputImage)