function Iout = ConvolutionjI(Iin, B)
Iin=double(Iin);
B=double(B);
M = size(B, 1);
N = size(B, 2);
Iout = zeros(size(Iin, 1), size(Iin, 2));
for k = 1:size(Iin, 1)-M
    for l = 1:size(Iin, 2)-N
        for i = k:(k+M-1)
            for j = l:(l+N-1)
                Iout(k,l) = Iout(k,l) + Iin(i,j) * B(i-k+1, j-l+1);
            end
        end
    end
end

end
