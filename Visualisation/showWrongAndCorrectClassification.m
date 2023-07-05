function [] = showWrongAndCorrectClassification(comparison, images, classificationResult)

%We display 100 of the correctly classified images
figure();
count=0;
i=1;
title('Correct Classification')
while (count<100)&&(i<=length(comparison))
   
    if comparison(i)
        count=count+1;
        subplot(10,10,count)
        Im = reshape(images(i,:),27,18);
        % for unsegmented images, change Im to uint8(Im)
        imshow(Im)
    end
    i=i+1;
end


%We display 100 of the incorrectly classified images
figure();
count=0;
i=1;
title('Wrong Classification')
while (count<100)&&(i<=length(comparison))
    
    if ~comparison(i)
        count=count+1;
        subplot(10,10,count)
        Im = reshape(images(i,:),27,18);
        % for unsegmented images, change Im to uint8(Im)
        imshow(Im)
        title(num2str(classificationResult(i)))
    end
    i=i+1;
end