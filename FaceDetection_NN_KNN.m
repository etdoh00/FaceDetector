
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
    % to use feature extractors, use the
    % FaceDetection_NN_KNN_FeatureExtractors


% Training Method
    % Since NN is the same as KNN with k=1, this file can be used for both
    % methods.
        % for applying NN, just set the k-value to 1
        % for KNN (with k != 1), choose another k value (k > 1)
addpath .\KNN\

% Visualisation
    % in order to get a better understanding of the data, there will be
    % some visualisations created during the training and evaluation stage
addpath .\Visualisation\


% Evalaluation

    

%% training Stage

% Load labels and images of faces and non-faces for training
sampling=1;
[images labels]= loadFaceImages('face_train.cdataset', sampling);

% For this binary classification, we only deal with face and no face images
% labeled with -1 and 1 correspondingly
indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

images= [images(indexesFace,:); images(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

% For visualization purposes, we display the first 100 images and plot the
% 2D-PCA validation of the spread between the classes
showFirst100Images(images, labels);
show2DPCA(images, labels)

%Supervised training function takes the training data to infer a model
modelKNN = NNtraining(images, labels);


%% testing
% Load labels and images of faces and non-faces for training
[images labels]= loadFaceImages('face_test.cdataset',sampling);

indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

images= [images(indexesFace,:); images(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

%For each testing image, we obtain a prediction based on our trained model
for i=1:size(images,1)
    testnumber= images(i,:);
    %to apply NN, change the value of K to 1
    K = 1;          
    classificationResult(i,1) = KNNTesting(testnumber, modelKNN, K);
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
