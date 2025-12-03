function Lk = Feature_similarity(train_data,k)
%计算局部余弦相似度拉普拉斯
X = train_data;
[~,dim]=size(X);

for i=1:dim  %for each  data
    %Training: get top k-nearest neighbors
    test = X(:,i);
    for j=1:dim %compute the distance with every training data 
        train = X(:,j);
        %V=test-train;
        A = dot(train,test);
        B = (norm(train,2)*norm(test,2)); 
        if B==0
            B=1;
        end
        dist(i,j)=A/B;
    end  
end
[~ , ind] = sort(dist,2);
K = eye(dim);
for i = 1:dim
    for j = 1:k+1
        K(i,ind(i,end-j)) = dist(i,ind(i,end-j));
    end
end

D = diag(sum(K,2));
Lk = D-K;
end