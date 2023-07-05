function [] = showFirst100Images(images, labels)

figure
for i=1:100
    % for visualisation, the images need to be reshaped to 27x18 format
    Im = reshape(images(i,:),27,18);
    % for unsegmented images, change Im to uint8(Im)
    subplot(10,10,i), imshow(Im), title(['label: ',num2str(labels(i))]);
end
