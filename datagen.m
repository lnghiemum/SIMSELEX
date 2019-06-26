% Data generation code for simulation in Section 4
function data = datagen(seed,n,p,case_theta, sigma, sigma_star, type)
% seed: seed for random number generation (for replication)
% n : sample size
% p : number of variables
% case_theta: different configurations for the true coefficients
% sigma: variance of regression error term, use for linear regression
% sigma_star: measurement error variance is sigma_star* eye(p)
% type: 1 for linear, 2 for logistic, and 3 for survival
rng(seed);

mu = zeros(p,1);
SIGMA = zeros(p,p);
%% Generate X ~ N_p (0, SIGMA), where SIGMA_{ij} = 0.25^{|i-j|)
for i=1:p
    for j=1:p
        SIGMA(i,j) = 0.25^(abs(i-j));
    end
end   
X = mvnrnd(mu,SIGMA,n);

%% Case_theta: use case_theta=1 and 2 for linear regression, case_theta=1
% and 3 for logistic and survival 
if case_theta==1
    theta = [ones(1,5) zeros(1,p-5)]';
elseif case_theta==2
    theta = [(1:5).^(-1) zeros(1,p-5)]';
elseif case_theta==3
    theta = [linspace(2,1,5) zeros(1,p-5)]';
end 

%% Generate response y
switch type
    case 'gaussian'
        error = normrnd(0,sigma,n,1);
        y = X * theta + error;
    case 'binomial'
    y = zeros(n,1);
    logit = X * theta;
    for j=1:n
        pr = 1/(exp(-logit(j))+1);
        y(j) = binornd(1,pr);
    end  
    case 'cox'
     v = rand(n,1);
     lambda=0.01; rho=1; rateC=0.001;
     Tlat = (- log(v) ./ (lambda * exp(X * theta))).^(1 / rho);
     C=exprnd(1/rateC,n,1);
     time = min([Tlat'; C'])';
     status= Tlat<=C;
     y = [time status];
end
%% Generate measurement errors U and contaminated W
SIGMA_U = sigma_star^2.* eye(p);
U = mvnrnd(mu,SIGMA_U,n);
W = X + U;
%% Save the data and setting
data=struct('W',W,'y',y,'X',X,'sigma',sigma, 'sigma_star', sigma_star, 'theta',theta,'SIGMA_U',SIGMA_U);

end
