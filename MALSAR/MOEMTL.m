function [ BME ] = MOEMTL( trainData, trainLabel, categoricalColNum, testData, testLabel)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
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
    expert_num = categoricalColNum;
    train_category_index = cell([1 expert_num]);
    test_category_index = cell([1 expert_num]);
    category_name = cell([1 expert_num]);
    for t = 1 : expert_num
        train_category_index{t} = categorical(table2array(trainData(:,t)));
        test_category_index{t} = categorical(table2array(testData(:,t)));
        category_name{t} = categories(train_category_index{t});
    end
    trainData = table2array(trainData(:, expert_num + 1:end));
    
    %trainTarget = table2array(trainLabel);
    trainTarget = str2num(cell2mat(table2cell(trainLabel)));
    
    testData = table2array(testData(:,expert_num + 1:end));
    
    %testTarget = testLabel;
    testTarget = str2num(cell2mat(table2cell(testLabel)));
    
    %% Create BME
    N = size(trainTarget, 1);
    BME = BMECreate('NumExperts', expert_num , 'MaxIt', 1e6, 'EType', 'mtl', 'ENbf', 0.1, 'EKernel', 'linear', 'EKParam', 0.5, ...
    'GType', 'mlr', 'GNbf', 0.1, 'GKernel', 'linear', 'GKParam', 0.5, 'MinLogLikeChange', N/1e5);
    %% BME.Experts.Input
    EInput = cell([1, expert_num]);
    MTLTarget = cell([1, expert_num]); % mtl target
    
    
    %%tempTarget
    for i = 1 : expert_num
        task_num = length(category_name{i});
        X = cell([1 task_num]);
        Y = cell([1 task_num]);
        for t = 1: task_num
            isSelected = ismember(train_category_index{i}, category_name{i}(t));
            X{t} = trainData(isSelected, :);
            Y{t} = trainTarget(isSelected,:);
            Y{t}(Y{t} == 0) = -1;
        end
        EInput{i} = X;
        MTLTarget{i} = Y;
    end
    %BME.Experts.Category_name = category_name;
    Category_name = category_name;

    %BME.Experts.Category_index = train_category_index;
    ECategory_index = train_category_index;
    %% gating input
    GInput = trainData;
    %% hold out
    task_sample_size = size(testTarget, 1);
    hold_out_percentage = 0.3;
    Idx = randperm(task_sample_size) < task_sample_size * hold_out_percentage;
    
    holdoutData = testData(Idx,:);
    holdoutTarget = testTarget(Idx,:);
    holdout_category_index = cell([1 expert_num]);
    for t = 1 : expert_num
        holdout_category_index{t} = test_category_index{t}(Idx,:);
    end
    holdout_MTL_input = cell([1, expert_num]);
    holdout_MTL_target = cell([1, expert_num]);
    for i = 1 : expert_num
        task_num = length(category_name{i});
        X = cell([1 task_num]);
        Y = cell([1 task_num]);
        for t = 1: task_num
            isSelected = ismember(holdout_category_index{i}, category_name{i}(t));
            X{t} = holdoutData(isSelected, :);
            Y{t} = holdoutTarget(isSelected,:);
        end
        holdout_MTL_input{i} = X;
        holdout_MTL_target{i} = Y;
    end
    
    
    
    testData = testData(~Idx,:);
    testTarget = testTarget(~Idx,:);
    
    temp_category_index = cell([1 expert_num]);
    for t = 1 : expert_num
        temp_category_index{t} = test_category_index{t}(~Idx,:);
    end
    test_category_index = temp_category_index;
%     testData
%     testTarget
%     test_category_index
    %% MOE test input
    %%%%%% expert
    test_expert_input = cell([1, expert_num]);
    %test_expert_target = cell([1, expert_num]);

    for i = 1 : expert_num
        task_num = length(category_name{i});
        X = cell([1 task_num]);
        %Y = cell([1 task_num]);
        for t = 1: task_num
            isSelected = ismember(test_category_index{i}, category_name{i}(t));
            X{t} = testData(isSelected, :);
            %Y{t} = testTarget(isSelected,:);
        end
        test_expert_input{i} = X;
        %test_expert_target{i} = Y;
    end
    %%
    %test_expert_target;

    TestEInput = test_expert_input;
    %BME.Test.EInput = test_expert_input;
    TestECategory_index = test_category_index;
    %BME.Test.ECategory_index = test_category_index;

    %%%% gateing

    TestGInput = testData;
    %BME.Test.GInput = test_gatings_input;
    %% EParam
    EParam = cell([1 expert_num]);
    for t = 1:expert_num
        EParam{t} = 1e-8;
    end    
    BME.Experts.Param = EParam;
    
    %% BMEInit
    %% Initialize BME 
    BME = BMEInit(BME, EInput, Category_name, ECategory_index, EParam, GInput, TestEInput, TestECategory_index, TestGInput, holdout_MTL_input);
    
    %% initialize
    for i = 1:BME.NumExperts    
        BME = BMEExpertsTrain(MTLTarget, holdout_MTL_target, BME, i) ;
    end
    BME.Experts.Means = BMEExpertsMeans(BME.Experts.Input, BME.Experts.Weights, BME.Experts.Category_name, BME.Experts.Category_index);
    BME.Gatings.Posteriors = BMEGatingsPosterior(trainTarget, BME);
    %% jobtitle MTL
    testMeans = BMEExpertsMeans(BME.Test.EInput, BME.Experts.Weights, BME.Experts.Category_name, BME.Test.ECategory_index);
    
    
    
    %% MOE
    BME = BMETrain(BME, MTLTarget, trainTarget, testTarget, holdout_MTL_target);
    
    %% figure
%     testMeans = BMEExpertsMeans(BME.Test.EInput, BME.Experts.Weights, BME.Experts.Category_name, BME.Test.ECategory_index);
%     testGatingOutput = exp(BME.Test.GInput*BME.Gatings.Weights);
%     sumTestGatingOutput = sum(testGatingOutput, 2);
%     TestProb = sum(testMeans.* testGatingOutput, 2) ./ sumTestGatingOutput;
    cTree = table2array(readtable('decisionTreeResult.csv','ReadRowNames',true));
    %% figure
    hold on
    p1 = prec_rec(cTree(:, 1), cTree(:, 2), 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p1_corr = corr(cTree(:, 1) >= 0.5, cTree(:, 2));
    
    p2 = prec_rec(testMeans(:, 1), testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p2_corr = corr(testMeans(:, 1) >= 0.5, testTarget);
    p3 = prec_rec(testMeans(:, 2), testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p3_corr = corr(testMeans(:, 2) >= 0.5, testTarget);
    p4 = prec_rec(testMeans(:, 3), testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p4_corr = corr(testMeans(:, 3) >= 0.5, testTarget);
    p5 = prec_rec(testMeans(:, 4), testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p5_corr = corr(testMeans(:, 4) >= 0.5, testTarget);



    testProb = BME.Test.TestProb(:, end);
    p6 = prec_rec(testProb, testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    p6_corr = corr(testProb >= 0.5, testTarget);
    
%     rTree = table2array(readtable('regressionTreeResult.csv','ReadRowNames',true));
%     p1 = prec_rec(rTree(:, 1), rTree(:, 2), 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
%     p1_corr = corr(rTree(:, 1) >= 0.5, rTree(:, 2));
    
    hold off
    %['MOE&MTL' num2str(p3_corr)]
    %legend([p1 p2 p3 p4 p5 p6],'decisionTree', 'jobtitle', 'department','purtimeframe','industry', 'MOE&MTL');
    legend('decisionTree', 'jobtitle', 'department','purtimeframe','industry', 'MOE&MTL');
    
    %% coefficient
    c = {'decisionTree', 'jobtitle', 'department','purtimeframe','industry', 'MOE&MTL'};
    x = [p1_corr p2_corr p3_corr p4_corr p5_corr p6_corr];
    bar(x);
    set(gca, 'XTickLabel', c);
    set(gca,'ylim',[0.15 0.23]);
end

