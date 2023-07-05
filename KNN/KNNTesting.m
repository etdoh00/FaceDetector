function prediction = KNNTesting(testImage, modelKNN, k)
s1 = testImage;

for j=1:size(modelKNN.neighbours,1)
    s2=modelKNN.neighbours(j,:);
    dEuc(j)=EuclideanDistance(s2,s1);
end

[sorted_dEuc, sorted_indexes] = sort(dEuc);
winner_class = zeros(1,max(modelKNN.labels)+2)
for i = 1:k
    class = modelKNN.labels(sorted_indexes(i));
    winner_class(class+2) = winner_class(class + 2) + 1;
end
[y pos] = max(winner_class);
prediction = pos(1)-2;

end