function Objects = simpleNMS(Objects, threshold)


for i = 1:size(Objects,1)
    for j = 1:size(Objects,1)
        if i ~= j
            % calcualte rectangle intersection area. Formate: [x,y,width,height]
            intArea = rectint(Objects(i,1:4),Objects(j,1:4));
            % calculate union area 
            unionArea = Objects(i,3)*Objects(i,4)+Objects(j,3)*Objects(j,4)-intArea;
            if (intArea/unionArea) > threshold
                if (Objects(i,5) > Objects(j,5))
                    Objects(j,:) = zeros(1,size(Objects,2));
                else 
                    Objects(i,:) = zeros(1,size(Objects,2));
                end
            end
        end
    end
end

count = 1;
while count <= size(Objects, 1)
    if Objects(count,:) == zeros(size(Objects,2))
        Objects(count,:) = [];
    else 
        count=count+1;
    end
end

        