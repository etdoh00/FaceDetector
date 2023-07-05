function PredictedObjects = MultiscaleSlidingWindow(I, modelSVM, ScalerWidth,ScalerHeight,ScalerMin,ScalerMax)

ObjectsTemp = zeros(100,5);

% Implement Multiscale
for scaler=ScalerMin:ScalerMax
    ScaleWidthTemp = scaler*ScalerWidth;
    ScaleHeightTemp = scaler*ScalerHeight;

    % Iterate through rows
    for r = 1:ScalerHeight*2:size(I,1)
        for c = 1:ScalerWidth*2:size(I,2)
            if (c+ScaleWidthTemp-1 <= size(I,2)) && (r+ScaleHeightTemp-1 <= size(I,1))
                % crop image
                digitIm = I(r:r+ScaleHeightTemp-1, c:c+ScaleWidthTemp-1);
                % convert image to double & invert
                digitIm = im2double(digitIm);
                % resample image
                digitIm = imresize(digitIm, [27, 18]);
                % reshape digit into vector
                digitIm = reshape(digitIm,1, 27*18);
                % do the prediction
                [prediction, maxi] = SVMDetection(digitIm, modelSVM);
                
   
                % if face is predicted, create Object
                if (prediction == 1)
                    newObject = [c,r,ScaleWidthTemp, ScaleHeightTemp, maxi];
                    ObjectsTemp=[ObjectsTemp; newObject];
                end
            end
        end
    end
end

PredictedObjects = ObjectsTemp;
