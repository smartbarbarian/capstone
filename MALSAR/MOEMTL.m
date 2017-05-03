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
    'GType', 'mtl', 'GNbf', 0.1, 'GKernel', 'linear', 'GKParam', 0.5, 'MinLogLikeChange', N/1e5);
    %% BME.Experts.Input
    EInput = cell([1, expert_num]);
    MTLTarget = cell([1, expert_num]); % mtl target
    
    
    tempTarget
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
    %% Initialize BME using kmeans clustering
    BME = BMEInit(BME, EInput, Category_name, ECategory_index, EParam, GInput, TestEInput, TestECategory_index, TestGInput, holdout_MTL_input) ; 
    %% MOE
    BME = BMETrain(BME, MTLTarget, trainTarget, testTarget, holdout_MTL_target);
    
end

