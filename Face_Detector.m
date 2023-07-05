clear all
close all

% General information on how to use this Face Detector

% The Face Detector is able to detect faces on a given image by using a 
% Multi-Scale Sliding Window, NMS and our selected ML model.

% Model: 
    % First, we are loading our trained faceDetectorModel which we saved in the
    % FaceDetection-SVM or -(K)NN. This will be used for detecting the faces
        % To be able to apply it, we need to import some paths
        % Note: to use another model than SVM, adjust MultiscaleSlidingWindow
        % Note: When applying the model, it's usual name (e.g. modelSVM)
        % should be used
addpath .\SVM\
addpath .\SVM-KM\
addpath .\KNN\


% Image & Preprocessing
    % First, one of the given images is saved as ITemp. Afterwards, some
    % Proprocessing methods (HE, Segmentation, Edge Extraction) can be
    % applied.
addpath .\images\
addpath .\Preprocessing\


% Visualisation
    % The bounding boxes of the images will be visualized before and after
    % NMS
addpath .\Visualisation\

% Detector
    % To detect images we first apply an Multiscale Sliding Window to the
    % image, where we detect bounding boxes with a specific confidence rate. 
    % To refine the detected Objects and remove unwanted ones, we apply
    % NMS afterwards.
addpath .\Detector\

% Evaluation
    % we saved all face objects of the 4 test images inside matricies to 
    % evaluate our results.
    % The evaluateDetector method compares the detected objects tp the
    % true ones and returns based on that the TP, FP and FN.
    % Furthermore, Recall, Precision and F-Measure are calculated.


%We load the classification model of our choice
load faceDetectorModel;


%Open testing image and do preprocessing.
% If you change the image here, change it also below for the evaluation of
% the detector
ITemp=imread('im1.jpg');

%% Preprocessing

% apply histogram equalisation
ITemp= histeq(ITemp);

% apply segmentation
%gray = graythresh(ITemp);
%ITemp = imbinarize(ITemp,gray);

% apply edge extraction
%ITemp = edge_extraction(ITemp);

I = ITemp;


%% Sliding Window & NMS

% Multiscale Sliding Window
% Input Parameters:
    % Image
    % Model
    % Width and Height Scale (here: 2x3)
    % Minimum & Maximum Scaler for Multiscale (e.g. for 7: 14x21 windows)
PredictedObjects = MultiscaleSlidingWindow(I, modelSVM,2,3,7,9);

% Show the results of sliding window
showDetectionResult(I,PredictedObjects);

% Apply NMS
NMSObjects = simpleNMS(PredictedObjects, 0.15);

% Show the results of NMS
showDetectionResult(I,NMSObjects);



%% Evaluation

% Solution objects contain all faces : (x,y,width,hight)
im1 = [7 20 18 27; 24 13 18 24; 48 26 15 23; 63 11 19 26;
    82 25 18 22; 99 11 17 23; 118 21 20 28];

im2 = [6 10 15 23; 31 10 15 23; 56 10 15 23; 81 10 15 23; 108 10 15 23;
       6 37 15 23; 31 37 15 23; 56 37 15 23; 81 37 15 23; 108 37 15 23; 
       6 69 15 23; 31 69 15 23; 56 69 15 23; 81 69 15 23; 108 69 15 23];

im3 = [16 36 18 19; 40 46 16 22; 67 23 17 20; 64 49 16 21;
    93 31 17 24; 114 59 17 19; 90 74 18 18; 27 77 18 19];

im4 = [18 188 19 23; 25 115 19 24; 51 87 18 24; 63 71 18 20; 42 52 20 25; 
    67 32 15 20; 80 143 19 26; 90 117 18 23; 109 88 17 23; 104 44 18 23; 
    114 186 19 23; 141 155 18 21; 147 120 16 21; 141 84 17 21; 158 52 17 20; 
    141 31 15 18;176 166 16 22; 208 180 21 22; 211 113 15 19; 194 85 14 18; 
    208 62 15 20; 207 23 16 20; 267 25 24 19; 235 45 15 16; 255 67 16 19; 
    257 97 17 22; 273 144 19 20; 303 187 19 21; 304 128 18 19; 323 91 15 21; 
    293 77 13 18; 284 51 13 19; 322 26 14 16; 335 59 14 19; 364 77 15 21; 
    387 114 16 22; 364 132 17 21; 338 150 16 20; 375 191 17 15; 390 148 18 21;
    451 183 19 24; 472 153 18 23; 450 129 18 20; 441 95 16 18; 409 64 14 21; 
    381 64 15 17; 380 33 16 20; 405 33 15 18; 438 46 15 18; 456 68 15 20; 
    466 35 15 19; 487 43 16 21;  521 76 18 23; 495 96 17 23; 513 117 17 22; 
    550 147 19 20; 531 195 19 23];

% Evaluate Detector returns TP, PF, and FN. Add the correct solution image
% as parameter.
[TP, FP, FN] = evaluateDetector(im1, NMSObjects)

% Calculate (recall Sensitivity, hit rate)
Recall = TP / (TP+FN) 

% Calculate Precision 
Precision = TP / (TP+FP)

% Calcluate F-Measure (F1) 
F_measure = (2*TP) / (2*TP + FN + FP)

