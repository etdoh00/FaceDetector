function [] = show2DPCA(images, labels, model)

[U,S,X_reduce] = pca(images,3);
imean=mean(images,1);
X_reduce=(images-ones(size(images,1),1)*imean)*U(:,1:3);

figure, hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;
for i=min(labels):max(labels)
    count = count+1;
    indexes = find (labels == i);
    plot3(X_reduce(indexes,1),X_reduce(indexes,2),X_reduce(indexes,3),colours(count,:)) 
end

% If this is called by FaceDetection_SVM, it also displayes the Support
% Vectors
if nargin>2
    hold on
    
    %transformation to the full image to the best 3 dimensions
    imean=mean(images,1);
    xsup_pca=(model.xsup-ones(size(model.xsup,1),1)*imean)*U(:,1:3);
    
    % plot support vectors
    h=plot3(xsup_pca(:,1),xsup_pca(:,2),xsup_pca(:,3),'go');
    set(h,'lineWidth',3)
    legend('No Face', 'Face', 'Support Vectors');
else 
    legend('No Face', 'Face');
end
