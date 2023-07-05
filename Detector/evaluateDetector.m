function [TP, FP, FN] = evaluateDetector(solutionObjects, NMSObjects)
% Create empty lists for TP and FP; save solutionObjects as false Negatives
TP=[];
FP=[];
FN = solutionObjects;
% loop through all detected Objects
for i=1:size(NMSObjects,1)
    % before the comparisons the NNSobject is assumed to be false
    correctpred = false; 
    % loop through all the solution objects
    for j=1:size(solutionObjects,1)
        % calcualte rectangle intersection area between both objects
        intArea = rectint(NMSObjects(i,1:4),solutionObjects(j,1:4));
        % check if the overlap is more than 50 %
        if (intArea/(solutionObjects(j,3).*solutionObjects(j,4))>0.5)
            % now the NMS object is known to be a correct prediction
            correctpred = true;
            % since this solution object is predicted correctly, remove it
            % but so that there won't be issues with further loops.
            FN(j,:) = zeros(1,size(FN,2));
        end
    end
    % correctly predicted objects are added to TP, the others to FP
    if (correctpred)
        TP = [TP; NMSObjects(i,:)];
    else 
        FP = [FP; NMSObjects(i,:)];
    end
end

% in the end hard delete the identified faces from the FN array.
count = 1;
while count <= size(FN, 1)
    if FN(count,:) == zeros(size(FN,2))
        FN(count,:) = [];
    else 
        count=count+1;
    end
end

% calculate the number of TP, FP, and FN
TP = size(TP,1);
FP = size(FP,1);
FN = size(FN,1);
