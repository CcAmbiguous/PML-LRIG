function [D] = PML_LR(X,Y,k)
% k 近邻数量
[n, ~] = size(X);
[~, q] = size(Y);

%% k-means聚类
idx = kmeans(X,q);
R = dummyvar(idx);
A = R*R';
%% 重构样本相关性
distance = pdist2(X, X, 'squaredeuclidean');
[near_sample, ind] = sort(distance,2);
N = zeros(n);
segma = sum(near_sample(:,2))/n;
S = exp(-distance/(segma^2));
for i=1:n
    for j = 1:k+1
        N(i,ind(i,j)) = 1;
    end
end
%% Optimization
S_new = N.*(S+A);
D = Y;
Q = distance;
max_iter = 5;
for tt=1:max_iter
 
    %% update S
    S = S.*(N.*S_new)./(N.*N.*(A+S) + Q +eps);
    
    %% update S_new
    S_new = S_new.*(N.*(A+S)+D*(Y')/k)./(S_new + S_new*Y*(Y')/(k^2)+eps);

    %% update D
    D = S_new*Y/k;
    D(D>1) = 1;
    D(Y==0)=0;

end
end

