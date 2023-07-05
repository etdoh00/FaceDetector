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
    % file, from which you load the images.
    % images. 
        % you can use the 'face_test.cdataset' and 'face_train.cdataset'
        % or face_train_half.cdataset and face_test_half.cdataset
addpath .\Train_Test_Splits\

% Feature Extractors
    % to use different feature extractors you can uncomment the
    % FeatureExtractors in the for-loop below loading the test AND train images
        % we implemented HOG and Gabor feature extractor
addpath .\Feature_Extractors\
       
% Classification Method
    % Since NN is the same as KNN with k=1, this file can be used for both
    % methods.
        % for applying NN, just set the k-value for the training to 1
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

% Uncomment exactly 1 out of GABOR and HOG. Make sure you apply the same to the test data
for i = 1:size(images, 1)
   Im = reshape(images(i,:),27,18);
   %features(i,:) = gabor_feature_vector(Im); % GABOR
   features(i,:) = hog_feature_vector(Im);  % HOG
end

% For this binary classification, we only deal with face and no face images
% labeled with -1 and 1 correspondingly
indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

features= [features(indexesFace,:); features(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

% For visualization purposes, we display the first 100 images
showFirst100Images(images, labels);

%Supervised training function takes the training data to infer a model
modelKNN = NNtraining(features, labels);


%% testing
% Load labels and images of faces and non-faces for training
[images labels]= loadFaceImages('face_test.cdataset',sampling);

% Uncomment exactly 1 out of GABOR and HOG. Make sure you applied the same to the train data
for i = 1:size(images, 1)
   Im = reshape(images(i,:),27,18);
   %features(i,:) = gabor_feature_vector(Im); % GABOR
   features(i,:) = hog_feature_vector(Im);  % HOG
end

indexesNoFace = find (labels == -1);
indexesFace = find (labels == 1);

features= [features(indexesFace,:); features(indexesNoFace,:)];
labels= [labels(indexesFace); labels(indexesNoFace)];

%For each testing image, we obtain a prediction based on our trained model
for i=1:size(features,1)
    testnumber= features(i,:);
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
