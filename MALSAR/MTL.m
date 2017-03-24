addpath('./MALSAR/functions/Lasso/'); % load function
addpath('./MALSAR/functions/SRMTL/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('./MALSAR/functions/joint_feature_learning/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('../ADASYN_upd2');%load adasyn(extension of SMOTE)
addpath('../prec_rec/prec_rec/')%load PRcruvr

data = readtable('rawdata.csv','ReadRowNames',true);



% 1 is new, 0 is existing
new = data(data.NewOrExisting == 1, :);
existing = data(data.NewOrExisting == 0, :);

%option2
SMB = table2array(new(strcmp(new.option2, 'SMB'), 1:66));
Enterprise = table2array(new(strcmp(new.option2, 'Enterprise'), 1:66));
%Unknown = new(strcmp(new.option2, 'Unknown'), 1:66);

%option4
Laggards = table2array(existing(strcmp(existing.option4, 'Laggards'), 1:66));
Hitters = table2array(existing(strcmp(existing.option4,'Heavy Hitters') | strcmp(existing.option4, 'Potentials'), 1:66));

X = {SMB(:, 1:65) Enterprise(:, 1:65) Laggards(:, 1:65) Hitters(:, 1:65)};
Y = {SMB(:, 66) Enterprise(:, 66) Laggards(:, 66) Hitters(:, 66)};

% preprocessing data
for t = 1: length(X)
    X{t} = zscore(X{t});                  % normalization
    X{t} = [X{t} ones(size(X{t}, 1), 1)]; % add bias. 
end



% split data into training and testing.
training_percent = 0.7;
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);


%move this step to crossvalidation
%because in cv test don't need balance but train need 
% 
% balance data   
% X_balance = cell(size(X_tr));
% Y_balance = cell(size(Y_tr));
% for t = 1: length(X)
%     X_majority = X_tr{t}(Y_tr{t}(:,1) == - 1, :);
%     Y_majority = Y_tr{t}(Y_tr{t}(:,1) == - 1, :);
%     [X_minority, Y_minority] = ADASYN(X_tr{t}, Y_tr{t}, [], [], [], true);
%     X_balance{t} = [X_minority; X_majority];
%     Y_balance{t} = [Y_minority; Y_majority];
% end



% the function used for evaluation.
eval_func_str = 'eval_MTL_mse';
higher_better = true;  % mse is lower the better.


% cross validation fold
cv_fold = 5;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.




% model parameter range
param_range = [0.0000001 0.000001 0.00001 0.0001 0.001];

fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation1Param...
    ( X_tr, Y_tr, 'Logistic_L21', opts, param_range, cv_fold, eval_func_str, higher_better);

%disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter 
[W, c, funcVal] = Logistic_L21(X_tr, Y_tr, best_param, opts);


Y_prob = cell(size(Y_te));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%lasso%%%%%%%%%%

param_range = [1 10 100 200 500 1000 2000]; %lamda

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.


fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation1Param...
    ( X_tr, Y_tr, 'Logistic_Lasso', opts, param_range, cv_fold, eval_func_str, higher_better);

%disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter 
[W, c, funcVal] = Logistic_L21(X_tr, Y_tr, best_param, opts);



%%%%%%%%%%%%%%%%%%%%%%%%%
for t = 1: length(X)
    Y_prob{t} = glmval(W(:, t), X_te{t}, 'logit','constant','off');
    prec_rec(Y_prob{t}, Y_te{t}, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1);
end



% 
% % show final performance
% eval_func = str2func(eval_func_str);
% final_performance = eval_func(Y_te, X_te, W);
% fprintf('Performance on test data: %.4f\n', final_performance);



% 
% 
% d = size(X{1}, 2);  % dimensionality.
% 
% lambda = [200 :300: 1500];
% % 
% sparsity = zeros(length(lambda), 1);
% log_lam  = log(lambda);
% 
% for i = 1: length(lambda)
%     [W funcVal] = Least_L21(X, Y, lambda(i), opts);
%     % set the solution as the next initial point. 
%     % this gives better efficiency. 
%     opts.init = 1;
%     opts.W0 = W;
%     sparsity(i) = nnz(sum(W,2 )==0)/d;
% end
% 
% % draw figure
% h = figure;
% plot(log_lam, sparsity);
% xlabel('log(\rho_1)')
% ylabel('Row Sparsity of Model (Percentage of All-Zero Columns)')
% title('Row Sparsity of Predictive Model when Changing Regularization Parameter');
% set(gca,'FontSize',12);
% print('-dpdf', '-r100', 'LeastL21Exp');
