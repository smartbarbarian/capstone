% clear;
% clc;
% close;
cd '/Users/palejustice/Documents/capstone/MALSAR';
addpath('./MALSAR/functions/Lasso/'); % load function
addpath('./MALSAR/functions/SRMTL/'); % load function
addpath('./MALSAR/functions/low_rank/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('./examples/train_and_test/'); % load train and test tools
addpath('./MALSAR/functions/joint_feature_learning/'); % load function
addpath('./MALSAR/utils/'); % load utilities
addpath('../ADASYN_upd2');%load adasyn(extension of SMOTE)
addpath('../prec_rec/prec_rec/')%load PRcruvr
addpath('../fastBayesianMixtureOfExperts-master')%load 
addpath('../SMOTE');%load SMOTE)
%%
data = readtable('XD_PCA_P_Simp.csv','ReadRowNames',true);

%% extract categories

JobTitle = categories(categorical(table2array(data(:,2))));
Department = categories(categorical(table2array(data(:,3))));
PURTimeframe = categories(categorical(table2array(data(:,4))));
Industry = categories(categorical(table2array(data(:,70))));


%%%% meaningless
AccountID2 = categories(categorical(table2array(data(:,69))));


category_name = {JobTitle, Department, PURTimeframe, Industry};
expert_num = length(category_name);
category_index = {categorical(table2array(data(:,2))), ...
    categorical(table2array(data(:,3))), ...
    categorical(table2array(data(:,4))), ...
    categorical(table2array(data(:,70)))};
%% newData without categories

date = datenum(table2array(data(:, 1)), 'yyyy/mm/dd');
label = double(strcmp(data.option1,'1'));
row = height(data);
employeeNum = zeros([row 1]);
for t = 1:row
    employeeNum(t, 1) = str2double(data{t, 78});
end
%temp = table2array(data(:, [5:68 71:75]));
temp = table2array(data(:, [5:7 9:68 71:75]));
newData = [date temp employeeNum];



%dataWithNum = data(~strcmp(data.option3,'NULL'), :);
%cat(1,dataWithNum{:, 78});
%data(:, 1) = datenum(table2array(data(:, 1)), 'yyyy/mm/dd')
%str2num


%% split data into training and testing.
training_percent = 0.7;

sample_size = size(newData, 1);
tSelIdx = randperm(sample_size) < sample_size * training_percent;
    
selIdx{t} = tSelIdx;

trainData = newData(tSelIdx,:);
trainTarget = label(tSelIdx,:);
testData = newData(~tSelIdx,:);
testTarget = label(~tSelIdx,:);
train_category_index = cell(size(category_index));
test_category_index = cell(size(category_index));
for t = 1 : expert_num
    train_category_index{t} = category_index{t}(tSelIdx,:);
    test_category_index{t} = category_index{t}(~tSelIdx,:);
end


%% partion and input
%% Create BME
BME = BMECreate('NumExperts', expert_num , 'MaxIt', 20, 'EType', 'mtl', 'ENbf', 0.1, 'EKernel', 'linear', 'EKParam', 0.5, ...
    'GType', 'mlr', 'GNbf', 0.1, 'GKernel', 'linear', 'GKParam', 0.5);


%% decision tree
% simpely convert catergory into numeric 
% %sbmatlab = table(trainTarget, train_category_index{1}, train_category_index{2}, train_category_index{3}, train_category_index{4});
% 
% tree_X = table(train_category_index{1}, train_category_index{2}, train_category_index{3}, train_category_index{4});
% 
% tc = fitctree(tree_X, trainTarget, 'AlgorithmForCategorical', 'Exact','CrossVal','on','MaxNumSplits',16);
% ens = fitcensemble(tree_X, trainTarget);
% et = compact(ens);
% 
% tc = fitctree(tree_X, trainTarget,'CrossVal','on');
% %
% tc = fitctree(tree_X, trainTarget, 'AlgorithmForCategorical', 'Exact');
% ctc = compact(tc);
% %
% view(tc,'Mode','graph');
% 
% %CutType,'categorical'



%% BME.Gatings.Input



gatings_mu = zeros([1 size(trainData, 2)]);
gatings_sigma = zeros([1 size(trainData, 2)]);
gatings_input = zeros(size(trainData, 2));
%%%because this column has missing value; 

for j = 1:size(trainData, 2)
    isNan = isnan(trainData(:, j));
    [gatings_input(~isNan, j), gatings_mu(j), gatings_sigma(j)] = zscore(trainData(~isNan,j));
end

% isNan = isnan(trainData(:, 70));
% [gatings_input(~isNan, 70), gatings_mu(70), gatings_sigma(70)] = zscore(trainData(~isNan,70));
% trainData(isNan, 70) = 0;
% [gatings_input(:, 1:69), gatings_mu(1:69), gatings_sigma(1:69)]  = zscore(trainData(:, 1:69));
% 

%BME.Experts.Input = gatings_input;

%gatings_input = [ones(size(testData, 1), 1) gatings_input]; % add bias. 
GInput = gatings_input;

%BME.Gatings.Input = gatings_input;





%% BME.Experts.Input
EInput = cell([1, expert_num]);
MTLTarget = cell([1, expert_num]); % mtl target

experts_mu = cell([1, expert_num]);
experts_sigma = cell([1, expert_num]);

for i = 1 : expert_num
    task_num = length(category_name{i});
    X = cell([1 task_num]);
    Y = cell([1 task_num]);
    mu = cell([1 task_num]);
    sigma = cell([1 task_num]);
    for t = 1: task_num
        isSelected = ismember(train_category_index{i}, category_name{i}(t));
        
        selected = trainData(isSelected, :);
        mu{t} = zeros([1 size(selected, 2)]);
        sigma{t} = zeros([1 size(selected, 2)]);
        %%%because this column has missing value; 
        for j = 1:size(selected, 2)
            isNan = isnan(selected(:, j));
            [selected(~isNan, j), mu{t}(j), sigma{t}(j)] = zscore(selected(~isNan,j));
            selected(isNan, j) = 0;
        end
%         isNan = isnan(selected(:, 70));
%         [selected(~isNan, 70), mu{t}(70), sigma{t}(70)] = zscore(selected(~isNan,70));
%         selected(isNan, 70) = 0;
%         [selected(:, 1:69), mu{t}(1:69), sigma{t}(1:69)]  = zscore(selected(:, 1:69));

        %X{t} = selected;
        X{t} = selected;
        Y{t} = trainTarget(isSelected,:);
    end
    experts_mu{i} = mu;
    experts_sigma{i} = sigma;
    EInput{i} = X;
    MTLTarget{i} = Y;
end
%BME.Experts.Alpha = 1;


%BME.Experts.Input = Input;




%BME.Experts.Category_name = category_name;
Category_name = category_name;

%BME.Experts.Category_index = train_category_index;
ECategory_index = train_category_index;


%% MOE test input
%%%%%% expert
test_expert_input = cell([1, expert_num]);
test_expert_target = cell([1, expert_num]);

for i = 1 : expert_num
    task_num = length(category_name{i});
    X = cell([1 task_num]);
    Y = cell([1 task_num]);
    mu = experts_mu{i};
    sigma = experts_sigma{i};
    for t = 1: task_num
        isSelected = ismember(test_category_index{i}, category_name{i}(t));
        
        selected = testData(isSelected, :);
        %%%because this column has missing value; 
        for j = 1:size(selected, 2)
            isNan = isnan(selected(:, j));
            if sigma{t}(j) == 0 
                selected(~isNan, j) = (selected(~isNan,j) - gatings_mu(j)) / gatings_sigma(j);
            else    
                selected(~isNan, j) = (selected(~isNan,j) - mu{t}(j)) / sigma{t}(j);
            end    
            selected(isNan, j) = 0;
        end
        X{t} = selected;
        Y{t} = testTarget(isSelected,:);
    end
    test_expert_input{i} = X;
    test_expert_target{i} = Y;
end
%%
test_expert_target;

TestEInput = test_expert_input;
%BME.Test.EInput = test_expert_input;
TestECategory_index = test_category_index;
%BME.Test.ECategory_index = test_category_index;

%%%% gateing


test_gatings_input = zeros(size(testData, 2));
%%%because this column has missing value; 
for j = 1:size(testData, 2)
    isNan = isnan(testData(:, j));
    test_gatings_input(~isNan, j) = (testData(~isNan,j) - gatings_mu(j)) / gatings_sigma(j);  
end

% isNan = isnan(testData(:, 70));
% test_gatings_input(~isNan, 70) = (testData(~isNan,70)- gatings_mu(70)) / gatings_sigma(70);
% %testData(isNan, 70) = 0;
% test_gatings_input(:, 1:69)  = testData(:, 1:69) - gatings_mu(1 : 69) / gatings_sigma(70);
% %test_gatings_input = [ones(size(testData, 1), 1) gatings_input]; % add bias. 

TestGInput = test_gatings_input;
%BME.Test.GInput = test_gatings_input;

%% Expert parameter
%EParam = zeros ([1 expert_num]);
%JobTitle/ Department/ PURTimeframe/ Industry
%you need to modify by you self
%%set parameter 
%the function used for evaluation.
eval_func_str = 'eval_MTL_matthews';
higher_better = true;  %  correlation is higher the better.

% cross validation fold
cv_fold = 5;

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.

%model parameter range
%param_range = [0.000001 0.00001 0.0001 0.001];
%param_range = [1e-9, 1e-8, 1e-7, 1e-6];
%%
% 9    11      7      5
%19.62 17.55  19.54  18.2
param_range = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7];
traceEParam = cell([1 expert_num]);
tracePerformMat = cell([1 expert_num]);
%crossvalidation
for i = 1 : expert_num
    X = EInput{i};
    Y = MTLTarget{i};
    fprintf('Perform model selection via cross validation: \n')
    [ best_param, perform_mat] = CrossValidation1Param...
        ( X, Y, 'Logistic_Trace', opts, param_range, cv_fold, eval_func_str, higher_better);
    traceEParam{i} = best_param;
    tracePerformMat{i} = perform_mat;
end
%%
% 5      7       5        4
% 19.57  17.4    19.57    18.23
param_range = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
EParam = cell([1 expert_num]);
PerformMat = cell([1 expert_num]);
%crossvalidation
for i = 1 : expert_num
    X = EInput{i};
    Y = MTLTarget{i};
    fprintf('Perform model selection via cross validation: \n')
    [ best_param, perform_mat] = CrossValidation1Param...
        ( X, Y, 'Logistic_Trace', opts, param_range, cv_fold, eval_func_str, higher_better);
    EParam{i} = best_param;
    PerformMat{i} = perform_mat;
end
%%
%EParam = {1e-05, 1e-07, 1e-05, 1e-04};
BME.Experts.Param = EParam;


%% BMEInit
%% Initialize BME using kmeans clustering
BME = BMEInit(BME, EInput, Category_name, ECategory_index, EParam, GInput, trainTarget, TestEInput, TestECategory_index, TestGInput) ; 


%% MOE

% w = warning('query','last');
% id = w.identifier;
% warning('off',id) ;

BME = BMETrain(BME, MTLTarget, trainTarget, testTarget) ;
% MTLTarget ----- mtl structured target
% trainTarget ----- normal target MTLTarget







%% test






% Y_sum = zeros([1 task_num]);
% for t = 1: task_num
%     Y_sum(t) = sum(Y{t});
% 
% end

% %% preprocessing data
% %aaa(~isnan(aaa(:,71)),71) = zscore(aaa(~isnan(aaa(:,71)),71))
% %
% for t = 1: length(X)
%     X{t} = zscore(X{t});                  % normalization
%     X{t} = [X{t} ones(size(X{t}, 1), 1)]; % add bias. 
% end




% [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);


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


% 
% %%set parameter 
% %the function used for evaluation.
% eval_func_str = 'eval_MTL_matthews';
% higher_better = true;  %  correlation is higher the better.
% 
% 
% % cross validation fold
% cv_fold = 5;
% 
% 
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%
% 
% %rng('default');     % reset random generator. Available from Matlab 2011.
% opts.init = 0;      % guess start point from data. 
% opts.tFlag = 1;     % terminate after relative objective value does not changes much.
% opts.tol = 10^-5;   % tolerance. 
% opts.maxIter = 1000; % maximum iteration number of optimization.
% 
% 
% 
% 
% %% model parameter range
% param_range = [0.00001 0.00005 0.0001];
% % crossvalidation
% fprintf('Perform model selection via cross validation: \n')
% [ best_param, perform_mat] = CrossValidation1Param...
%     ( X_tr, Y_tr, 'Logistic_L21', opts, param_range, cv_fold, eval_func_str, higher_better);
% 
% %disp(perform_mat) % show the performance for each parameter.
% 
% 
% %% build model using the optimal parameter 
% %% caculate the p_value of coefficient
% test_num = 10; % the num of product
% W_sum = cell([1 test_num]);
% X_tr_ba = cell([1 task_num]);
% Y_tr_ba = cell([1 task_num]);
% for t = 1: task_num
%     X_majority = X_tr{t}(Y_tr{t}(:,1) == 0, :);
%     Y_majority = Y_tr{t}(Y_tr{t}(:,1) == 0, :);
% 
%     [X_minority, Y_minority] = ADASYN(X_tr{t}, Y_tr{t}, [], [], [], true);
% 
%     X_tr_ba{t} = [X_minority; X_majority];
%     Y_tr_ba{t} = double([Y_minority; Y_majority]);
%     Y_tr_ba{t}(Y_tr_ba{t} == 0) = - 1;
% end
% %% test
% [W, C, funcVal] = Logistic_L21(X_tr_ba, Y_tr_ba, best_param, opts);
% cor_MTL2 = eval_MTL_matthews(Y_te, X_te, W, C);
% 
% %% p_value
% for t = 1:test_num
%     [W, C, funcVal] = Logistic_L21(X_tr_ba, Y_tr_ba, best_param, opts);
%     W_sum{t} = W;
% end
% 
% p_value = zeros(size(W));
% W_mean = zeros(size(W));
% [row, col] = size(W);
% for i = 1:row
%     for j = 1:col
%         temp = zeros([1 test_num]);
%         for t = 1:test_num
%             temp(t) = W_sum{t}(i,j);
%         end
%         mean_temp = mean(temp);
%         std_temp = std(temp);
%         W_mean(i, j) = mean_temp;
%         [h,p]= ztest(0, mean_temp, std_temp);
%         p_value(i, j) = p;
%     end
% end
% 
% 
% 
% 
% 
% 
% 
% %% prec_recall figure
% Y_prob = cell(size(Y_te));
% for t = 1: length(X)
%     Y_prob{t} = glmval([C(t); W(:, t)], X_te{t}, 'logit','constant','on');
%     %Y_prob{t} = glmval(W(:, t), X_te{t}, 'logit');
%     prec_rec(Y_prob{t}, Y_te{t}, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1);
% end
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% %%%%%%%%%%%lasso%%%%%%%%%%
% % 
% % param_range = [1 10 100 200 500 1000 2000]; %lamda
% % 
% % %rng('default');     % reset random generator. Available from Matlab 2011.
% % opts.init = 0;      % guess start point from data. 
% % opts.tFlag = 1;     % terminate after relative objective value does not changes much.
% % opts.tol = 10^-5;   % tolerance. 
% % opts.maxIter = 1500; % maximum iteration number of optimization.
% % 
% % 
% % fprintf('Perform model selection via cross validation: \n')
% % [ best_param, perform_mat] = CrossValidation1Param...
% %     ( X_tr, Y_tr, 'Logistic_Lasso', opts, param_range, cv_fold, eval_func_str, higher_better);
% % 
% % %disp(perform_mat) % show the performance for each parameter.
% % 
% % % build model using the optimal parameter 
% % [W, c, funcVal] = Logistic_L21(X_tr, Y_tr, best_param, opts);
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% % 
% % % show final performance
% % eval_func = str2func(eval_func_str);
% % final_performance = eval_func(Y_te, X_te, W);
% % fprintf('Performance on test data: %.4f\n', final_performance);
% 
% 
% 
% % 
% % 
% % d = size(X{1}, 2);  % dimensionality.
% % 
% % lambda = [200 :300: 1500];
% % % 
% % sparsity = zeros(length(lambda), 1);
% % log_lam  = log(lambda);
% % 
% % for i = 1: length(lambda)
% %     [W funcVal] = Least_L21(X, Y, lambda(i), opts);
% %     % set the solution as the next initial point. 
% %     % this gives better efficiency. 
% %     opts.init = 1;
% %     opts.W0 = W;
% %     sparsity(i) = nnz(sum(W,2 )==0)/d;
% % end
% % 
% % % draw figure
% % h = figure;
% % plot(log_lam, sparsity);
% % xlabel('log(\rho_1)')
% % ylabel('Row Sparsity of Model (Percentage of All-Zero Columns)')
% % title('Row Sparsity of Predictive Model when Changing Regularization Parameter');
% % set(gca,'FontSize',12);
% % print('-dpdf', '-r100', 'LeastL21Exp');