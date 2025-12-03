% Initialization
clear all
close all
% Fixed seed
rng('default')
addpath(genpath('datasets')); % Add path
addpath(genpath('function'));
addpath(genpath('metrics'));

% Import data set
dataname = 'emotions';
avg_cls = 3;% The amount of noise added
[pLabels,data,target] = addnoise(dataname,avg_cls);

%%               
opt.alpha = 1;                     
opt.beta = 1; 
opt.gamma = 0.1; 
opt.lambda = 1; 
opt.ratio  = 0.2;
opt.max_iter = 20;
%%
[N,d] = size(data);
[~,c] = size(target);
indices = crossvalind('Kfold', 1:N ,10);  % Dividing the data set
                
for round = 1:10
    ht = round*10;
    fprintf('%.1f%%\n',ht)
    test_idxs = (indices == round);                       
    train_idxs = ~test_idxs;                       
    train_data = data(train_idxs,:);                                           
    train_target = pLabels(train_idxs,:); 
    true_target = target(train_idxs,:); 
    test_data = data(test_idxs,:);                                          
    test_target = target(test_idxs,:);
                    
                      
    % pre-processing 归一化                                       
    [train_data, settings]=mapminmax(train_data');                                        
    test_data=mapminmax('apply',test_data',settings);                                          
    train_data(isnan(train_data))=0;                                           
    test_data(isnan(test_data))=0;                                           
    train_data=train_data';                                           
    test_data=test_data';                                             
    X = train_data;
    Xt = test_data;
    Y = train_target;
    Yt = test_target;
    [D] = PML_LR(X,Y,5);

    %High dimensional kernel mapping   
    [K,Kt] = Kernel_mapping(X',Xt');                       
    X = K';   
    Xt = Kt';
    % 
    %% training
    tic;
    model = PML_IG(X,D,opt);
    time(round) = toc;
    %% testing
    [HammingLoss(round),RankingLoss(round),OneError(round),Coverage(round),AveragePrecision(round),~] = PML_test(Xt,Yt,model);
end

fprintf('%s,avg_cls=%.1f,lambda=%.5f,alpha=%.5f,beta=%.5f,gamma=%.5f,ratio=%.1f\n HammingLoss=%.3f±%.3f\n RankingLoss=%.3f±%.3f\n OneError=%.3f±%.3f\n Coverage=%.3f±%.3f\n AveragePrecision=%.3f±%.3f\n', ...
    dataname,avg_cls,opt.lambda,opt.alpha,opt.beta,opt.gamma,opt.ratio,mean(HammingLoss),std(HammingLoss),mean(RankingLoss),std(RankingLoss),mean(OneError),std(OneError),mean(Coverage),std(Coverage),mean(AveragePrecision),std(AveragePrecision));


fprintf('time=%.3f±%.3f\n',mean(time),std(time));    




