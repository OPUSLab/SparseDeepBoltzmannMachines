%
% Source code for "Training deep Boltzmann networks with sparse Ising machines"
% by Shaila Niazi, Shuvro Chowdhury, Navid Anjum Aadit, Masoud Mohseni, Yao Qin & Kerem Y. Camsari
% Nature Electronics volume 7, pages610–619 (2024)
%
% Date: July 2023
%

% This code performs the training of MNIST/100 or Full MNIST based on the selection
% This CPU code is expensive for sampling, the results shown in the paper are done with FPGA

clc; clearvars; close all;

warning off;
rng(12345);                  % Changing rng will change the index of visible and label bits (also works similarly)

full_mnist = 0;             % 0 for MNIST/100, 1 for full MNIST

% load  p-bit colors 
colormap = readmatrix('colorMap_4264.csv');
required_colors = length(unique(colormap));

Groups = cell(1,required_colors);
for k = 1:required_colors
    Groups{k} = find(colormap==k);
end

% Load adjacency matrix (extracted from D-Wave)
load('JJ_4264.mat')
W = sparse(W);

num_pbits = length(W); 													% number of total pbits

% % randomize indices
index = randperm(num_pbits);
index_visible = index(1:784);
index_sticker1 = index(785:794);
index_sticker2 = index(795:804);
index_sticker3 = index(805:814);
index_sticker4 = index(815:824);
index_sticker5 = index(825:834);

beta = 1;															    % inverse temperature

if(full_mnist)
    % parameters
    num_images	=	60000; 												% number of training samples/truth table lines
    size_batch	=	50; 												%number of images in a batch
    num_batch	=	num_images/size_batch;
    num_samples =	500; 											    % number of sweeps to be read
    NL	=	500; 														% number of learning steps or epochs
    num_samples_to_wait_neg = 200;										% number of sweeps to wait between two reads in the negative phase
    num_samples_to_wait_pos = num_samples_to_wait_neg/size_batch;       % number of sweeps to wait between two reads in the positive phase
    num_samples_to_wait = 100;                                          % number of sweeps to wait between two reads during inference
    num_images_to_test = 10000;                                         % all images from the test set

    eps = 0.003*ones(1,NL); 											% learning rate
    lambda = 0;															% regularization
    al_pha = 0.6; 														% momentum

    % % Loading Full MNIST dataset
    oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
    filenameImagesTrain = 'train-images-idx3-ubyte.gz';
    filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';

    XTrain = processImagesMNIST(filenameImagesTrain);
    YTrain = processLabelsMNIST(filenameLabelsTrain);

    y_label = double(string(YTrain)); % converting categorical to numeric values
    t_image = zeros(num_images,length(index_visible));

    % % binarize the images
    for r = 1:num_images
        t_image(r,:) = reshape(2*ceil(XTrain(:,:,1,r))-1,1,[]);
    end
else
    num_images = 100; % number of training samples/truth table lines
    size_batch=10; %number of images in a batch
    num_batch=num_images/size_batch;
    num_samples =size_batch*50;                                                % number of sweeps to read

    num_samples_to_wait_neg = 10;										      % number of sweeps to wait between two reads in the negative phase
    num_samples_to_wait_pos = ceil(num_samples_to_wait_neg/size_batch);       % number of sweeps to wait between two reads in the positive phase
    num_samples_to_wait = 100;
    num_images_to_test = 100;

    NL=500; % number of learning steps
    eps=linspace(0.06,0.006,NL);
    lambda=0.000;
    al_pha = 0.6;

    %% Loading MNIST/100 dataset from MATLAB
    [XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
    y_label=double(string(YTrain)); % converting categorical to numeric values

    % collecting indices for all 10 digits
    label_index=zeros(500,10);

    for digits=1:10
        label_index(:,digits)=find(y_label==digits-1);
    end

    num_each_digits=num_images/10;
    sorted_images=label_index(1:num_each_digits,1:10);
    sorted_images=sorted_images(:);

    t_image=zeros(num_images,length(index_visible));
    for r=1:num_images
        xx=ceil(XTrain(:,:,1,sorted_images(r)));
        bb=xx(:)';
        t_image(sorted_images(r),:)=2*bb-1;
    end
    t_image= t_image(sorted_images,:);
end

% % 10 visible clamped bits for label/sticker
ts=1000;
ts1=ts*[+1 -1 -1 -1 -1 -1 -1 -1 -1 -1];
ts2=ts*[-1 +1 -1 -1 -1 -1 -1 -1 -1 -1];
ts3=ts*[-1 -1 +1 -1 -1 -1 -1 -1 -1 -1];
ts4=ts*[-1 -1 -1 +1 -1 -1 -1 -1 -1 -1];
ts5=ts*[-1 -1 -1 -1 +1 -1 -1 -1 -1 -1];
ts6=ts*[-1 -1 -1 -1 -1 +1 -1 -1 -1 -1];
ts7=ts*[-1 -1 -1 -1 -1 -1 +1 -1 -1 -1];
ts8=ts*[-1 -1 -1 -1 -1 -1 -1 +1 -1 -1];
ts9=ts*[-1 -1 -1 -1 -1 -1 -1 -1 +1 -1];
ts10=ts*[-1 -1 -1 -1 -1 -1 -1 -1 -1 +1];

t_sticker=[ts1; ts2; ts3; ts4; ts5; ts6; ts7; ts8; ts9; ts10];

% % Initializing to Gaussian
J0 = W.*(0.01*randn([num_pbits,num_pbits]));
J=(J0+J0')/2;
J(J==diag(J))=0;

% Initialization of bias
hbias=(randi([-1,1],1,num_pbits)); 
pr_images = mean((1+t_image)/2);
hbias(index_visible) = changem(log(pr_images./(1-pr_images)),[-100,100],[-Inf,+Inf]);

hc=zeros(1,num_pbits);

if(full_mnist)
    % % Clamping h to truth table values
    hclamp_train = zeros(num_images,num_pbits);

    for rr=1:num_images
        hclamp_train(rr,index_visible) = 1000*t_image(rr,:);
        hclamp_train(rr,index_sticker1) = t_sticker((y_label(rr,:)+1),:);
        hclamp_train(rr,index_sticker2) = t_sticker((y_label(rr,:)+1),:);
        hclamp_train(rr,index_sticker3) = t_sticker((y_label(rr,:)+1),:);
        hclamp_train(rr,index_sticker4) = t_sticker((y_label(rr,:)+1),:);
        hclamp_train(rr,index_sticker5) = t_sticker((y_label(rr,:)+1),:);
    end

else
    % % Clamping h to truth table values
    hclamp_train = zeros(num_images,num_pbits);

    for rr=1:num_images
        hclamp_train(rr,index_visible)=1000*t_image(rr,:);
        hclamp_train(rr,index_sticker1)=t_sticker((y_label(sorted_images(rr),:)+1),:);
        hclamp_train(rr,index_sticker2)=t_sticker((y_label(sorted_images(rr),:)+1),:);
        hclamp_train(rr,index_sticker3)=t_sticker((y_label(sorted_images(rr),:)+1),:);
        hclamp_train(rr,index_sticker4)=t_sticker((y_label(sorted_images(rr),:)+1),:);
        hclamp_train(rr,index_sticker5)=t_sticker((y_label(sorted_images(rr),:)+1),:);
    end
end

ss = cell(size_batch,1);
s = zeros(num_samples/size_batch, num_pbits);
s_model = zeros(num_samples, num_pbits);

delta_w_previous=0;
delta_bias_previous=0;

x = zeros(num_pbits,1);
s_temp = sign(2*rand(num_pbits,1)-1);


% placeholders for testing
s_temp2 = double(sign(2*rand(num_pbits,1)-1));
x2 = (zeros(num_pbits,1, 'double'));
s_test = zeros(num_samples,num_pbits);
accuracy_train=zeros(1,NL);

% randomize images over all epochs
rnd_tt = randperm(num_images);
for a = 1:NL
    rnd_tt(a,:) = randperm(num_images);
end

% tic
for ll = 1:NL
    tic_epoch = tic;
    for jj = 1:(num_batch)
        tic_pos = tic;
        hc_batch = hclamp_train(rnd_tt(((jj-1)*size_batch+1):jj*size_batch),:);
        J_bipolar = sparse(J);

        for kk = 1:size_batch
            hc = hbias+hc_batch(kk,:);
            h_bipolar = hc;
            for k = 1:num_samples/size_batch
                % to emulate FPGA
                for klc = 1:1:num_samples_to_wait_pos
                    for ijk = 1:1:required_colors
                        x(Groups{ijk}) =  beta*(J_bipolar(Groups{ijk},:)*s_temp+h_bipolar(Groups{ijk})');
                        s_temp(Groups{ijk}) = sign(tanh(x(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
                    end
                end
                s(k,:) = s_temp';
            end
            ss{kk,:} = s;
        end

        mm = vertcat(ss{:});
        data = sparse(((mm'*mm)/num_samples).*W);
        biasdata = mean(mm);
        time_pos = toc(tic_pos);

        tic_neg = tic;

        % % Calculating <mimj> model (Negative Phase)
        h_bipolar = hbias;
        for k = 1:num_samples
            % to emulate FPGA
            for klc = 1:1:num_samples_to_wait_neg
                for ijk = 1:1:required_colors
                    x(Groups{ijk}) =  beta*(J_bipolar(Groups{ijk},:)*s_temp+h_bipolar(Groups{ijk})');
                    s_temp(Groups{ijk}) = sign(tanh(x(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
                end
            end
            s_model(k,:) = s_temp';
        end
        mm1 = s_model;
        modelcorr = sparse(((mm1'*mm1)/num_samples).*W);
        biasmodel = mean(mm1);
        time_neg = toc(tic_neg);

        tic_update = tic;

        delta_w = eps(ll)*(data - modelcorr) + al_pha*(delta_w_previous);
        delta_bias = eps(ll)*(biasdata - biasmodel) + al_pha*(delta_bias_previous);

        J = J + delta_w - eps(ll)*lambda*J;
        hbias = hbias + delta_bias - eps(ll)*lambda.*hbias;
        delta_w_previous = delta_w;
        delta_bias_previous = delta_bias;
        time_update = toc(tic_update);
    end
    fprintf('Epoch %d: Pos Phase: %.2fs, Neg Phase: %.2fs, Updates: %.2fs\n', ...
        ll, time_pos, time_neg, time_update);
    time_epoch = toc(tic_epoch);
    fprintf('Total epoch time: %.2fs\n', time_epoch);

    % To save the J and h for the very first update
%{     if (ll==1 && jj==1)
          
          fprintf('running ll=1 and jj= %0.6f\n',jj)
            Je1=J;
            he1=hbias;

          fname_J1 = append('Outputs/Je1_',num2str(jj),'.mat');
          fname_h1 = append('Outputs/he1_',num2str(jj),'.mat');
          save(fname_J1,'Je1')
          save(fname_h1,'he1')
        
    end %}

    Jout = (J);
    hout = (hbias);
    fname_J = append('Outputs/Jout_',num2str(ll),'.mat');    
    fname_h = append('Outputs/hout_',num2str(ll),'.mat');
    save(fname_J,'Jout')                                    % saving the new Jout in the 'Outputs' directory
    save(fname_h,'hout')


    if(mod(ll,5)==0)
        J_test = sparse(Jout);  
        h_test = hout;

        hclamp_test1= zeros(num_images_to_test,num_pbits);
        h1_train= zeros(num_images_to_test,num_pbits);

        for rr=1:num_images_to_test
            hclamp_test1(rr,index_visible)=1000*t_image(rr,:);
            h1_train(rr,:)=h_test+hclamp_test1(rr,:);
        end


        count_wrong_train = 0;

        for n_train = randperm(num_images_to_test)
            h_test2 = h1_train(n_train,:);
            for k = 1:num_samples
                for klc = 1:num_samples_to_wait
                    for ijk = 1:1:required_colors
                        x2(Groups{ijk}) =  beta*(J_test(Groups{ijk},:)*s_temp2+h_test2(Groups{ijk})');
                        s_temp2(Groups{ijk}) = sign(tanh(x2(Groups{ijk}))-2*rand(length(Groups{ijk}),1)+1);
                    end
                end
                s_test(k,:) = s_temp2';
            end
            mm3 = (1+s_test)/2;                   % converted to binary format

            mean1_mm=(mean(mm3(:,index_sticker1)))';
            mean2_mm=(mean(mm3(:,index_sticker2)))';
            mean3_mm=(mean(mm3(:,index_sticker3)))';
            mean4_mm=(mean(mm3(:,index_sticker4)))';
            mean5_mm=(mean(mm3(:,index_sticker5)))';

            mean_mm=(mean1_mm+mean2_mm+mean3_mm+mean4_mm+mean5_mm)/5;
            [mean_max,index_of_maxmm]=max(mean_mm);

            if(full_mnist)
                train_index = n_train;
            else
                train_index = sorted_images(n_train);
            end

            if(y_label(train_index)==index_of_maxmm-1)
                % 		fprintf('\n\n\n n_train = %d, index_of_image = %d, detecting!!!, showing %d and also detecting %d\n\n\n',n_train,train_index,y_label(train_index),index_of_maxmm-1);
            else
                count_wrong_train=count_wrong_train+1;

                % 		fprintf('n_train = %d, index_of_image = %d, can not detect correctly, showing %d, but detecting %d\n',n_train,train_index,y_label(train_index),index_of_maxmm-1);

            end

        end
        % end

        accuracy_train(ll)=((num_images_to_test-count_wrong_train)/num_images_to_test); %accuracy in percentage
        fprintf('accuracy_train is %d.\n',accuracy_train(ll)*100);
        
    end
end
%toc

save Jout.mat Jout
save hout.mat hout

save index_visible.mat index_visible
save index_sticker1.mat index_sticker1
save index_sticker2.mat index_sticker2
save index_sticker3.mat index_sticker3
save index_sticker4.mat index_sticker4
save index_sticker5.mat index_sticker5
