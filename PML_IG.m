function model = PML_IG(X,D,opt)

warning('off');
rng('default')

lambda = opt.lambda;
alpha = opt.alpha;
beta = opt.beta;
gamma = opt.gamma;
ratio = opt.ratio;
max_iter = opt.max_iter;

model = [];

[~,d]=size(X);
[~,q]=size(D);

%% Training
% 定义优化参数
l = ceil(d*ratio);
H = randn(d,l);
M = randn(l,q);
C = cosineSimilarity(D', D');
F = D;
P = eye(q);
G = D*D';
Lk = Feature_similarity(X,5);
temp = pinv(X'*X);
miniLossMargin = 1e-4;%收敛性判断阈值
Lc1 = diag(sum(C,1)) - C;
Lc2 = diag(sum(C,2)) - C;
delta = norm(F,'fro')^2/norm(D,'fro')^2;
loss(1) = norm(X*H*M-D,'fro')^2+lambda*(norm(M,'fro')^2+norm(H,'fro')^2)+alpha*(norm(D*P-F,'fro')^2+norm(F*(F')-delta*G,'fro')^2)+beta*norm(D-F*C,'fro')^2+2*gamma*(trace(H'*Lk*H)+2*trace(M*Lc1*M'+M*Lc2*M'));
for ii = 1:max_iter
    
    %
    Lc1 = diag(sum(C,1)) - C;
    Lc2 = diag(sum(C,2)) - C;
    

    %% update H
    Ha = gamma*temp*Lk ;
    Hb = M*(M')+ lambda*eye(l);
    Hc = temp*(X')*D*(M');

    H = sylvester(Ha,Hb,Hc);
    clear Ha Hb Hc

    %% update M
    Ma = (X*H)'*(X*H);
    Mb = lambda*eye(q) + gamma*(Lc1 + Lc2);
    Mc = (X*H)'*D;

    M = sylvester(Ma,Mb,Mc);
    clear Ma Mb Mc

    %% update P
    [U,~,V] = svd(F'*D);
    P = V*(U');
    clear U V

    %% update F
    FA = (4*alpha*F*(F')*F + 2*alpha*F + 2*beta*F*(C*C'));
    FB = 4*alpha*delta*G*F + 2*alpha*D*P + 2*beta*D*(C');
    seita = max(0,F-D);
    FC = seita.*D;
    F = F.*FB./FA - FC./FA;

    clear FA FB FC

    delta = norm(F,'fro')^2/norm(D,'fro')^2;

    %% update C
    O = pdist2(M', M', 'squaredeuclidean');
    B = 2*beta*F'*D;
    A = 2*beta*(F)'*F*C + gamma*O + eps;
     
    C = C.*B./A;
    clear A B

    
    

    
    %%
    loss(ii+1) = norm(X*H*M-D,'fro')^2+lambda*norm(M,'fro')^2+lambda*norm(H,'fro')^2+alpha*(norm(D*P-F,'fro')^2+norm(F*(F')-delta*G,'fro')^2)+beta*norm(D-F*C,'fro')^2+gamma*(trace(H'*Lk*H)+2*trace(M*Lc1*M'+M*Lc2*M'));
    if ii>5
        temp_loss = (loss(ii+1) - loss(ii))/loss(ii); 
        if temp_loss<miniLossMargin
            % ccc = 1;
            break;%
        end
    end
    

end
model.loss = loss;
model.W = H*M;
model.F = F;
model.C = C;
end