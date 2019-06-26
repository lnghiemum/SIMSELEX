% The function performs variable selection step for SIMSELEX
function est = group_fit(M, lambda)
% M: matrix of estimates M_{ij} = \theta_j(\lambda_i} after the simulation
% step
% gamma = sequence of added noise
X = [ones(length(lambda),1) lambda' lambda'.^2];
m = size(X,2); q=size(M,2);
Y = M;
% Tolerance
delta = 10^-5;

K = X'*Y;
for j = 1:q
    normM(j) = norm(K(:,j),2);
end
lambda_max = max(normM)/size(X,1);
lambda_min = 0.001*lambda_max;

gamma_seq = logspace(lambda_max, lambda_min);
% choosing tuning parameter from cross validation 
CV = zeros(length(gamma_seq),1);
for j=1:length(gamma_seq)
    error_fold = zeros(length(lambda),1);
    for i = 1:size(Y,1)
        % leave one out cross validation
        trainY = Y;
        trainY(i,:) = [];
        trainX = X;
        trainX(i,:) = [];
        B = pg(trainX,trainY,delta,gamma_seq(j));
        error_fold(i) = norm(Y(i,:)-X(i,:)*B,2);
    end
    CV(j) = mean(error_fold);
end
% Using one-stanndard-error rule
f = find(CV<=min(CV)+std(CV,1));
gamma=gamma_seq(max(f));
est = pg(X,Y,delta,gamma);
end

function B = pg(X,Y,delta, gamma)
% delta = threshold; default = 10^-10;
m = size(X,2); q=size(Y,2);
B = inv(X'*X)*X'*Y;
L = max(eig(X'*X/size(X,1)));
nu = (1/L)/20;
conv = 0; % control convergence
count = 0;
d = 0;
while conv==0
    d_prev=d;
    B_prev = B;
    for i = 1:q
        omega = B(:,i) + nu*X'*(Y(:,i)-X*B(:,i));
        B(:,i) = max([0 1-gamma*nu/norm(omega,2)])*omega;
    end
    count = count + 1;
    d = norm(B-B_prev,2);
    % If there is an increase in the magnitude of d, start again but with a
    % smaller step size \nu
    if and(count>1, abs(d)>abs(d_prev))
        B = inv(X'*X)*X'*Y;
        conv=0;
        nu=nu/5;
    else   
    conv = norm(B-B_prev,2)<delta; 
    end
end
end 

