clear;
clc;
close;
cd '/Users/palejustice/Documents/capstone/MALSAR';
addpath('./MALSAR/functions/Lasso/'); % load function
addpath('./MALSAR/functions/SRMTL/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('./examples/train_and_test/'); % load train and test tools
addpath('./MALSAR/functions/joint_feature_learning/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('../ADASYN_upd2');%load adasyn(extension of SMOTE)
addpath('../prec_rec/prec_rec/')%load PRcruvr
%%
data = readtable('XD_PCA_P_Simp.csv','ReadRowNames',true);

%% extract categories

JobTitle = categories(categorical(table2array(data(:,2))));
Department = categories(categorical(table2array(data(:,3))));
PURTimeframe = categories(categorical(table2array(data(:,4))));
Industry = categories(categorical(table2array(data(:,70))));


AccountID2 = categories(categorical(table2array(data(:,69))));

%% newData without categories

date = datenum(table2array(data(:, 1)), 'yyyy/mm/dd');
label = double(strcmp(data.option1,'1'));
row = height(data);
employeeNum = zeros([row 1]);

for t = 1:row
    employeeNum(t, 1) = str2double(data{t, 78});
end
temp = table2array(data(:, [5:68 71:75]));
%temp = table2array(data(:, [5:7 9:68 71:75]));
newData = [date temp employeeNum];


%dataWithNum = data(~strcmp(data.option3,'NULL'), :);
%cat(1,dataWithNum{:, 78});
%data(:, 1) = datenum(table2array(data(:, 1)), 'yyyy/mm/dd')
%str2num
%% partion
task_num = length(PURTimeframe);
X = cell([1 task_num]);
Y = cell([1 task_num]);
for t = 1: task_num
    isSelected = strcmp(data.PURTimeframe, PURTimeframe(t));
    
    selected = newData(isSelected, :);
    
    isNan = isnan(selected(:, 71));
    selected(~isNan, 71) = zscore(selected(~isNan,71));
    selected(isNan, 71) = 0;
    selected(:, 1:70) = zscore(selected(:, 1:70));
    
    %X{t} = selected;
    X{t} = selected(:,[1:4 6:71]);
    Y{t} = label(isSelected,:);

end

%% test
Y_sum = zeros([1 task_num]);
for t = 1: task_num
    Y_sum(t) = sum(Y{t});

end

% %% preprocessing data
% %aaa(~isnan(aaa(:,71)),71) = zscore(aaa(~isnan(aaa(:,71)),71))
% %
% for t = 1: length(X)
%     X{t} = zscore(X{t});                  % normalization
%     X{t} = [X{t} ones(size(X{t}, 1), 1)]; % add bias. 
% end



%% split data into training and testing.
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



%%set parameter 
%the function used for evaluation.
eval_func_str = 'eval_MTL_matthews';
higher_better = true;  %  correlation is higher the better.


% cross validation fold
cv_fold = 5;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.




%% model parameter range
param_range = [0.00001 0.00005 0.0001];
% crossvalidation
fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation1Param...
    ( X_tr, Y_tr, 'Logistic_L21', opts, param_range, cv_fold, eval_func_str, higher_better);

%disp(perform_mat) % show the performance for each parameter.


%% build model using the optimal parameter 
%% caculate the p_value of coefficient
test_num = 10; % the num of product
W_sum = cell([1 test_num]);
X_tr_ba = cell([1 task_num]);
Y_tr_ba = cell([1 task_num]);
for t = 1: task_num
    X_majority = X_tr{t}(Y_tr{t}(:,1) == 0, :);
    Y_majority = Y_tr{t}(Y_tr{t}(:,1) == 0, :);

    [X_minority, Y_minority] = ADASYN(X_tr{t}, Y_tr{t}, [], [], [], true);

    X_tr_ba{t} = [X_minority; X_majority];
    Y_tr_ba{t} = double([Y_minority; Y_majority]);
    Y_tr_ba{t}(Y_tr_ba{t} == 0) = - 1;
end
%% test
[W, C, funcVal] = Logistic_L21(X_tr_ba, Y_tr_ba, best_param, opts);
cor_MTL2 = eval_MTL_matthews(Y_te, X_te, W, C);

%% p_value
for t = 1:test_num
    [W, C, funcVal] = Logistic_L21(X_tr_ba, Y_tr_ba, best_param, opts);
    W_sum{t} = W;
end

p_value = zeros(size(W));
W_mean = zeros(size(W));
[row, col] = size(W);
for i = 1:row
    for j = 1:col
        temp = zeros([1 test_num]);
        for t = 1:test_num
            temp(t) = W_sum{t}(i,j);
        end
        mean_temp = mean(temp);
        std_temp = std(temp);
        W_mean(i, j) = mean_temp;
        [h,p]= ztest(0, mean_temp, std_temp);
        p_value(i, j) = p;
    end
end







%% prec_recall figure
Y_prob = cell(size(Y_te));
for t = 1: length(X)
    Y_prob{t} = glmval([C(t); W(:, t)], X_te{t}, 'logit','constant','on');
    %Y_prob{t} = glmval(W(:, t), X_te{t}, 'logit');
    prec_rec(Y_prob{t}, Y_te{t}, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%lasso%%%%%%%%%%
% 
% param_range = [1 10 100 200 500 1000 2000]; %lamda
% 
% %rng('default');     % reset random generator. Available from Matlab 2011.
% opts.init = 0;      % guess start point from data. 
% opts.tFlag = 1;     % terminate after relative objective value does not changes much.
% opts.tol = 10^-5;   % tolerance. 
% opts.maxIter = 1500; % maximum iteration number of optimization.
% 
% 
% fprintf('Perform model selection via cross validation: \n')
% [ best_param, perform_mat] = CrossValidation1Param...
%     ( X_tr, Y_tr, 'Logistic_Lasso', opts, param_range, cv_fold, eval_func_str, higher_better);
% 
% %disp(perform_mat) % show the performance for each parameter.
% 
% % build model using the optimal parameter 
% [W, c, funcVal] = Logistic_L21(X_tr, Y_tr, best_param, opts);



%%%%%%%%%%%%%%%%%%%%%%%%%




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