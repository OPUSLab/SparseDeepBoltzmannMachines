clc;clearvars;close all;
% rng(12345);

% load adjacency and colormap
load('JJ_4264.mat')
colormap1 = readmatrix('colorMap_4264.csv');
required_colors = length(unique(colormap1));

Groups = cell(1,required_colors);
for k = 1:required_colors
    Groups{k} = find(colormap1==k);
end

NM=length(W); % number of total pbits
num_pbits=NM;

%sample image to clamp the labels of it
image_index= 169; % Change it as required


num_sweeps = 10000;  % number of sweeps


beta = [0:0.125:5]; % annealing schedule



% % Loading MNIST dataset
oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);
XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

y_label = double(string(YTrain)); 							% converting categorical to numeric values
y_label_test = double(string(YTest));


x=ceil(XTrain(:,:,1,image_index)); % from training dataset
% x=ceil(XTest(:,:,1,image_index)); % uncomment if you want the image from test set

figure(1)
colormap gray
imagesc(x)

bb=x(:)'; % Flatten x1 values and taking as row vector

t=2*bb-1;


load Jout_100.mat
load hout_100.mat

load index_visible.mat
load index_sticker1.mat
load index_sticker2.mat
load index_sticker3.mat
load index_sticker4.mat
load index_sticker5.mat

index_sticker=[index_sticker1; index_sticker2; index_sticker3; index_sticker4; index_sticker5];

J3=Jout;
h3=hout;



% % Clamping h to labels value
label=y_label(image_index);
% label=y_label_test(image_index); % uncomment if you want the image from test set

hclamp= zeros(1,NM);

hclamp(index_sticker)=-1000;
hclamp(index_sticker(:,label+1))=1000;


h1=h3+hclamp;

% save h1.mat h1
J_bipolar = sparse(J3);
h_bipolar = h1;

s = zeros(length(beta)*num_sweeps, num_pbits);
x = zeros(NM,1);
s_temp = 2*rand(NM,1)-1;

tic


for kk= 1:length(beta)


    fprintf('running \x3B2 = %0.6f\n',beta(kk))

    for k=1:num_sweeps

        for ijk = 1:1:required_colors
            x(Groups{ijk}) =  beta(kk)*(J_bipolar(Groups{ijk},:)*s_temp+h_bipolar(Groups{ijk})');
            s_temp(Groups{ijk}) = sign(tanh(x(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
        end
        s ((kk-1)*num_sweeps+k,:)= s_temp';

    end


end

toc

mm3=s;

% energy calculation to find the best energy sweep
%  Energy = zeros(1,length(beta)*num_sweeps);
%  for ii=1:length(beta)*num_sweeps
%      m1=2*s(ii,:)-1;
%      Energy(ii)=  -(m1*Jout/2*m1'+ m1*hout');
%  end

%  [min_Energy, I]= min(Energy);

%  figure(6)
%  plot(Energy)
%  nn=mm3(I,index_visible); % reading from all beta, best energy sweep

nn=mm3(end,index_visible); % reading only at last beta, single sweep

B=reshape(nn,[28,28]); %reshaping to 28 by 28 matrix
figure(2)
colormap gray
f1=imagesc(B)

