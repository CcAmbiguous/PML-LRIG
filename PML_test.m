function [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,MicroF1] = PML_test(test_data,test_target,model)
[num_test,~]=size(test_data);
[~,num_class]=size(test_target);

W = model.W;
Xt = test_data;
Outputs = Xt*W;
% [Outputs,~] = mapminmax(Outputs,0,1);
threshold = 0.5;%%阈值
Pre_Labels = zeros(num_test,num_class);
Pre_Labels(Outputs>=threshold)=1;
Pre_Labels(Outputs<threshold)=0;
HammingLoss=Hamming_loss(Pre_Labels,test_target);
RankingLoss=Ranking_loss(Outputs',test_target');
OneError=One_error(Outputs',test_target');
Coverage=coverage(Outputs',test_target');
AveragePrecision=Average_precision(Outputs',test_target');
MicroF1 = Micro_F1(Pre_Labels',test_target');

end

