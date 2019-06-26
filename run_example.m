% Sample code
data = datagen(2,100,300,1, 0.25, 0.15, 'gaussian');
y = data.y;
W = data.W;
SIGMA_U=data.SIGMA_U;
est = computeSIMSELEX(W,y,SIGMA_U, 'gaussian', 20, 5);
% The first term of each estimate is intercept
Naive = est.Naive;
SIMSELEX = est.SIMSELEX;
