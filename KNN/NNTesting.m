function prediction = NNTesting(testImage, modelNN)
s1 = testImage;

for i=1:size(modelNN.neighbours,1)
    s2=modelNN.neighbours(i,:);
    eucl(i)=EuclideanDistance(s1,s2);
end

[value,index] = min(eucl);

prediction = modelNN.labels(index);

end