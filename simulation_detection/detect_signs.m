
%% Train Faster R-CNN Vehicle Detector

load('rcnnStopSigns.mat','rcnn');   


% Read test image
%testImage = imread('stopSignTest.jpg');
testImage = imread('./data/s6.jpg');
%testImage = imread('./data/000015.png');

tic
% Detect stop signs
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128);

toc


% Display the detection results
[score, idx] = max(score);

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage)

