function Iout = edge_extraction(Iin)

%Standard
Bj=[-1 1];
Bi=[-1;1];

%Previtt
%Bj=[-1 0 1; -1 0 1; -1 0 1];
%Bi=[-1 -1 -1; 0 0 0; 1 1 1];

%Sobel
%Bj= [-1 0 1; -2 0 2; -1 0 1];
%Bi= [-1 -2 -1; 0 0 0; 1 2 1];

%Roberts
%Bj=[1 0; 0 -1];
%Bi=[0 1; -1 0];


HGradient = myconvolution(Iin, Bi);
VGradient = myconvolution(Iin, Bj);

ITemp = sqrt((HGradient.^2) + (VGradient.^2));

Iout = imbinarize(uint8(ITemp), 'adaptive', 'Sensitivity',0.2);

