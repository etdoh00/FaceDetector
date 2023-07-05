clear all
close all
% General information on how to use this script

% Preprocessing
    % to try out different preprocessing techniques, have a look at 
    % loadFaceImages.m in the Preprocessing folder where you can select:
        % Histogram Equalisation, Segmentation and Edge Extraction
        % Note: to display unsegmented images, add uint8 in the visualisation
        % functions
addpath .\Preprocessing\

% Data Augmentation
    % to acitvate/deactivate data augmentation, set the flag in
    % loadFaceImages in the Preprocessing folder. It creates variations
    % by mirroring, flipping, and displacements
        % We added other techniques like small rotations, shear and scale
        % using the imageDataAugmenter from the Deeplearning Toolbox

% Train-Test-Data-Splits
    % to change the type of Test-Train-Data-Splits, you need to change the
    % respective line in this FaceDetection script, where you load the
    % images
        % you can use the face_test.cdataset and face_train.cdataset
        % or face_train_half.cdataset and face_test_half.cdataset
addpath .\Train_Test_Splits\

% Feature Extractors
    % to use different feature extractors you can uncomment the sections
    % below loading train and test data
        % if you don't uncomment the for-loops, no feature extractions will
        % be applied
        % if you uncomment the for-loops and either HOG or Gabor the
        % respective technique will be applied
addpath .\Feature_Extractors\

% Training Method
    % This file is for applying the Support Vector Machine
addpath .\SVM\
addpath .\SVM-KM\

% Visualisation
    % in order to get a better understanding of the data, there will be
    % some visualisations created during the training and evaluation stage
addpath .\Visualisation\

% Face Detector
    % To use your created model in the face detector, you can save it as
    % faceDetectorModel after you trained it.

% Evaluation


%% training Stage

% Load labels and images of faces and non-faces for training
sampling=1;
[images labels]= loadFaceImages('face_train_half.cdataset', sampling);

% For this binary classification, we only deal with face and no face images
% labeled with -1 and 1 correspondingly
indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

images= [images(indexesFace,:); images(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

%Supervised training function that takes the examples and infers a model
modelSVM = SVMtraining(images, labels);
save faceDetectorModel modelSVM;

% For visualization purposes, we display the first 100 images and plot the
% 2D-PCA validation of the spread between the classes
showFirst100Images(images, labels);
show2DPCA(images, labels, modelSVM);

%% testing
% Load labels and images of faces and non-faces for training
[images labels]= loadFaceImages('face_test_half.cdataset',sampling);

indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

images= [images(indexesFace,:); images(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

%For each testing image, we obtain a prediction based on our trained model
for i=1:size(images,1)
    testnumber= images(i,:);
    classificationResult(i,1) = SVMTesting(testnumber, modelSVM);
end


%% Evaluation

% Compare predicted classification from ML algorithm against the reallabel
comparison = (labels==classificationResult);

% Identify correctly classified samples / total number of tested samples
Accuracy = sum(comparison)/length(comparison)

%We display 100 of the Wrongly and correctly classified images
showWrongAndCorrectClassification(comparison, images, classificationResult)

figure()

confusionmatrix=confusionchart(labels,classificationResult);
TN = confusionmatrix.NormalizedValues(1,1);
FP = confusionmatrix.NormalizedValues(1,2);
FN = confusionmatrix.NormalizedValues(2,1);
TP = confusionmatrix.NormalizedValues(2,2);
totalOccurence = sum(confusionmatrix.NormalizedValues,"all");
%Error rate 
Error_Rate = (FN+FP)/ totalOccurence

%Recall 
Recall = TP/(TP+FP)

%Precision
Precision = TP/(TP+FP)

%Specifictiy 
Specifictiy = TN / (TN+FP)

%F1
F1_Score = 2*TP/(2*TP+FN+FP)

%False alarm rate
False_Alarm_Rate = 1-Specifictiy
