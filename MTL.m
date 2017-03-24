
data = readtable('rawdata.csv','ReadRowNames',true);

% 1 is new, 0 is existing
new = data(data.NewOrExisting == 1, :);
existing = data(data.NewOrExisting == 0, :);

%option2
SMB = new(strcmp(new.option2, 'SMB'), 1:66);
Enterprise = new(strcmp(new.option2, 'Enterprise'), 1:66);
%Unknown = new(strcmp(new.option2, 'Unknown'), 1:66);

%option4
Laggards = existing(strcmp(existing.option4, 'Laggards'), 1:66);
Hitters = existing(strcmp(existing.option4,'Heavy Hitters') | strcmp(existing.option4, 'Potentials'), 1:66);

X = {SMB(:, 1:65) Enterprise(:, 1:65)  Laggards(:, 1:65) Hitters(:, 1:65)};
Y = {SMB(:, 66) Enterprise(:, 66)  Laggards(:, 66) Hitters(:, 66)};



addpath('./MALSAR/MALSAR/functions/Lasso/'); % load function
addpath('./MALSAR/MALSAR/functions/SRMTL/'); % load function
addpath('./MALSAR/MALSAR/utils/'); % load utilities
addpath('./MALSAR/MALSAR/functions/joint_feature_learning/'); % load function
addpath('./MALSAR/MALSAR/utils/'); % load utilities

d = size(X{1}, 2);  % dimensionality.

lambda = [200 :300: 1500];

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.

sparsity = zeros(length(lambda), 1);
log_lam  = log(lambda);

for i = 1: length(lambda)
    [W funcVal] = Least_L21(X, Y, lambda(i), opts);
    % set the solution as the next initial point. 
    % this gives better efficiency. 
    opts.init = 1;
    opts.W0 = W;
    sparsity(i) = nnz(sum(W,2 )==0)/d;
end

% draw figure
h = figure;
plot(log_lam, sparsity);
xlabel('log(\rho_1)')
ylabel('Row Sparsity of Model (Percentage of All-Zero Columns)')
title('Row Sparsity of Predictive Model when Changing Regularization Parameter');
set(gca,'FontSize',12);
print('-dpdf', '-r100', 'LeastL21Exp');
