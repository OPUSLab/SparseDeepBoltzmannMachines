%
% Source code for "Training deep Boltzmann networks with sparse Ising machines"
% by Shaila Niazi, Shuvro Chowdhury, Navid Anjum Aadit, Masoud Mohseni, Yao Qin & Kerem Y. Camsari
% Nature Electronics volume 7, pages610–619 (2024)
%
% Date: July 2023
%


% This code performs the testing after the training has been performed


clc; clearvars; close all;

warning off;
rng(12345);

% parameters

num_samples = 500;  										% number of sweeps to be read 
NT = num_samples; 
num_truth_tables = 10000; 									% number of training samples/truth table lines
num_images_test = 10000;									% number of images to be tested
num_samples_to_wait = 200;									% number of sweeps to wait between two reads   
beta = 1;											% inverse temperature
epoch = 100;											% number of epochs used for training
points = 1:1:epoch;										% epochs at which we want to measure accuracy


% load connectivity
load('JJ_4264.mat')

% load  p-bit colors
colormap = readmatrix('colorMap_4264.csv');
required_colors = length(unique(colormap));

Groups = cell(1,required_colors);
for k = 1:required_colors
    Groups{k} = find(colormap==k);
end

num_pbits = length(W); 											% number of total pbits


load index_visible.mat 
load index_sticker1.mat 
load index_sticker2.mat
load index_sticker3.mat
load index_sticker4.mat
load index_sticker5.mat

% % Loading full MNIST dataset

oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);
XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

y_label = double(string(YTrain)); 						% converting categorical to numeric values
y_label_test = double(string(YTest)); 						% converting categorical to numeric values

t_image_train=zeros(num_truth_tables,length(index_visible));
t_image_test=zeros(num_images_test,length(index_visible));

for rr = 1:num_truth_tables
    xx = ceil(XTrain(:,:,1,rr));
    bb = xx(:)';
    t_image_train(rr,:) = 2*bb-1;
end

for rr = 1:num_images_test 
    xx = ceil(XTest(:,:,1,rr));
    bb = xx(:)';
    t_image_test(rr,:) = 2*bb-1;
end

s = zeros(num_samples, num_pbits);
accuracy_train = zeros(1,length(points));
accuracy_test = zeros(1,length(points));

tic
for ll = points
	fname1 = sprintf("Outputs/Jout_%d",ll);
	fname2 = sprintf("Outputs/hout_%d",ll);

	load (fname1)
	load (fname2)

   	J_bipolar = sparse(Jout); 
   	h3 = hout;

% % Clamping h to truth table values to check the labels for classification

	hclamp_test1 = zeros(num_truth_tables,num_pbits);
	h1_train = zeros(num_truth_tables,num_pbits);

    for rr = 1:num_truth_tables
        hclamp_test1(rr,index_visible) = 1000*t_image_train(rr,:);
        h1_train(rr,:) = h3 + hclamp_test1(rr,:);
    end

	hclamp_test2 = zeros(num_images_test,num_pbits);
	h1_test = zeros(num_images_test,num_pbits);

    for rr = 1:num_images_test 
    	hclamp_test2(rr,index_visible) = 1000*t_image_test(rr,:);
    	h1_test(rr,:) = h3 + hclamp_test2(rr,:);
    end

	count_wrong_test = 0;
	count_wrong_train = 0;

	x = zeros(num_pbits,1);          
	s_temp = sign(2*rand(num_pbits,1)-1);  

	for n_train = randperm(num_truth_tables)
    	h_bipolar = h1_train(n_train,:);
    	for k = 1:num_samples
			for klc = 1:num_samples_to_wait
				for ijk = 1:1:required_colors
					x(Groups{ijk}) =  beta*(J_bipolar(Groups{ijk},:)*s_temp+h_bipolar(Groups{ijk})');
					s_temp(Groups{ijk}) = sign(tanh(x(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
				end
			end
			s(k,:) = s_temp'; 
    	end
		mm3 = (1+s)/2; % converting to binary

		mean1_mm = (mean(mm3(:,index_sticker1)))';
		mean2_mm = (mean(mm3(:,index_sticker2)))';
		mean3_mm = (mean(mm3(:,index_sticker3)))';
		mean4_mm = (mean(mm3(:,index_sticker4)))';
		mean5_mm = (mean(mm3(:,index_sticker5)))';
		mean_mm = (mean1_mm+mean2_mm+mean3_mm+mean4_mm+mean5_mm)/5;
		[mean_max,index_of_maxmm] = max(mean_mm);
		train_index = n_train; 

		if(y_label(train_index) == index_of_maxmm-1)
    	else 
        	count_wrong_train=count_wrong_train+1;
		end
	end

	accuracy_train(ll) = ((num_truth_tables-count_wrong_train)/num_truth_tables)*100;  %accuracy in percentage
	fprintf('accuracy_train is %d.\n',accuracy_train(ll));

	for n_test = 1:num_images_test
    	h_bipolar = h1_test(n_test,:);
    	for k=1:num_samples
			for klc = 1:1:num_samples_to_wait
				for ijk = 1:1:required_colors
					x(Groups{ijk}) =  beta*(J_bipolar(Groups{ijk},:)*s_temp+h_bipolar(Groups{ijk})');
					s_temp(Groups{ijk}) = sign(tanh(x(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
				end
			end
			s(k,:) = s_temp'; 
    	end   
		mm4 = (1+s)/2;  

		mean1_mm=(mean(mm4(:,index_sticker1)))';
		mean2_mm=(mean(mm4(:,index_sticker2)))';
		mean3_mm=(mean(mm4(:,index_sticker3)))';
		mean4_mm=(mean(mm4(:,index_sticker4)))';
		mean5_mm=(mean(mm4(:,index_sticker5)))';

		mean_mm_test=(mean1_mm+mean2_mm+mean3_mm+mean4_mm+mean5_mm)/5;

		[mean_max_test,index_of_maxmm_test]=max(mean_mm_test);

		test_index = n_test;

		if(y_label_test(test_index)==index_of_maxmm_test-1)
   		else 
        	count_wrong_test=count_wrong_test+1;
		end
	end

	accuracy_test(ll)=((num_images_test-count_wrong_test)/num_images_test)*100; %accuracy in percentage
        fprintf('accuracy_test is %d.\n',accuracy_test(ll)*100);


end

toc

figure(2)
plot(points,accuracy_train(points),'b','LineWidth',2.5)
hold on

ylim([0,100]);
xlabel('epoch');
ylabel('accuracy');

FontAxis = 20;
FontName = 'Arial';
plotSize=[5 7];
set(findall(gcf,'type','text'),'fontsize',FontAxis,'fontname',FontName);
set(gca,'FontSize',FontAxis,'fontname',FontName,'linewidth',0.3,'fontweight','bold');
 
hold on
plot(points,accuracy_test(points),'r','LineWidth',2.5)


grid on
box on
ax = gca;
ax.LineWidth = 1.5;
hold off

legend({'Training accuracy','Test accuracy'},'Location','northwest')

save testData.mat
