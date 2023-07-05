function [images labels] = loadFaceImages(filename,sampling)

if nargin<2
    sampling =1;
end

% this is a flag that allow you to activate/deactivate the data augmentation
% Data augmentation will increase the size of the dataset by created variations 
%(mirroring, flipping, displacements, rotations, shears, scales) of each given 
% image. This aims to produce more training images and, therefore, improve performance
augmented=1;

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

line1=fgetl(fp);
line2=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

images=[];
labels =[];
for im=1:sampling:numberOfImages
    
    label = fscanf(fp,'%d',1);
    
    labels = [labels; label];
    
    imfile = fscanf(fp,'%s',1);
    ITemp=imread(imfile);
    
    % apply histogram equalisation
    ITemp= histeq(ITemp);

    % apply segmentation
    gray = graythresh(ITemp);
    ITemp = imbinarize(ITemp,gray);

    % apply edge extraction
    %ITemp = edge_extraction(ITemp);

    I = ITemp;


    if size(I,3)>1
        I=rgb2gray(I);
   end
    vector = reshape(I,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    
    images= [images; vector];
    
    if augmented
        
        if label==1
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),1)
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            % rotation using imrotate with angles 3, 5, and 8 to right &
            % left
            
            Itemp = imrotate(I,3);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp = imrotate(I,5);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            Itemp = imrotate(I,8);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            Itemp = imrotate(I,352);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            Itemp = imrotate(I,355);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp = imrotate(I,358);
            Itemp = imresize(Itemp, [size(I,1), size(I,2)]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            % X and Y Shear using augmentor
            augmenter = imageDataAugmenter(RandXShear = [0 45]);
            Itemp = augment(augmenter, I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            augmenter = imageDataAugmenter(RandYShear = [0 45]);
            Itemp = augment(augmenter, I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];


            % X and Y scale using augmentor
            augmenter = imageDataAugmenter(RandXScale = [0.5 3]);
            Itemp = augment(augmenter, I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];

            augmenter = imageDataAugmenter(RandYScale = [0.5 2]);
            Itemp = augment(augmenter, I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];



            
            
        else
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =flipud(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =flipud(fliplr(I));
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];      
        end

    end
    
end

fclose(fp);

end
