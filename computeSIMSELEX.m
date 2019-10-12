% The function implements several estimators considered in the simulation
function output = computeSIMSELEX(W,y,SIGMA_U, type, B, numLambda)
% W: contaminated covariates  
% y: response
% sigma_U: measurement error variance
% type: 'gaussian' = linear, 'binomial' = logistic, 'cox' = survival
% B: number of replicates for each added measurement error level in the
% simulation step of SIMSELEX
% numLambda: number of added measurement error level in the simulation step
% of SIMSELEX
% This function requires installation of the glmnet package, which can be
% downloaded from https://web.stanford.edu/~hastie/glmnet_matlab/


n = length(y);
p = size(W,2);
% For cross-validation, using auc as performance measure for logistic model and deviance for everything else 
if strcmp(type,'binomial')==1
    type_error='auc';
else
    type_error='deviance';
end   
%% Naive Lasso estimates
% Cross-validation
naive_m = glmnet(W,y,type);
cv_n = cvglmnet(W,y,type,[],type_error);
bNAIVE = glmnetCoef(naive_m, cv_n.lambda_1se,true);
%% SIMEX estimates
%%% Simulation step
gamma_seq = linspace(0.2,2,numLambda);
M = zeros(length(gamma_seq),length(bNAIVE));
for i=1:length(gamma_seq)
    bsimex = zeros(B,length(bNAIVE));
    for b =1:B
        W_star = W + sqrt(gamma_seq(i))*mvnrnd(zeros(p,1),SIGMA_U, n);
        naive_simex = glmnet(W_star,y,type);
        cv_simex = cvglmnet(W_star,y,type,[],type_error);
        k=glmnetCoef(naive_simex, cv_simex.lambda_1se,false);
                bsimex(b,:)=k';
    end
    M(i,:) = mean(bsimex,1);
end    
newM = [bNAIVE';M];        

%%% Selection step based on group lasso
gamma = [0 gamma_seq];
B2 = group_fit(newM, gamma);
bSIMEX =[1 -1 1]*B2;

%%% Extrapolation step done on the selected variables
J = find(bSIMEX~=0);
MJ = newM(:,J);
bSIMSELEX = zeros(length(bNAIVE),1);
GAMMA = [ones(length(gamma),1) gamma' gamma'.^2];
BETA = GAMMA\MJ;
bSIMSELEX(J)=BETA' * [1 -1 1]';
output = struct('Naive',bNAIVE,'SIMSELEX',bSIMSELEX);

end

