function [ Weight ] = Grad_MTL( X, Y,  param, Grad)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    %% balance
    task_num = length(X);
    X_ba = cell([1 task_num]);
    Y_ba = cell([1 task_num]);
    for t = 1: task_num
        X_majority = X{t}(Y{t}(:,1) == 0, :);
        Y_majority = Y{t}(Y{t}(:,1) == 0, :);

        [X_minority, Y_minority] = ADASYN(X{t}, Y{t}, [], [], [], true);

        X_ba{t} = [X_minority; X_majority];
        Y_ba{t} = double([Y_minority; Y_majority]);
        Y_ba{t}(Y_ba{t} == 0) = - 1;
    end
    %% train
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%
    %rng('default');     % reset random generator. Available from Matlab 2011.
    opts.init = 0;      % guess start point from data. 
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-5;   % tolerance. 
    opts.maxIter = 1000; % maximum iteration number of optimization.
    [W, C, funcVal] = Logistic_L21(X_tr_ba, Y_tr_ba, param, opts, Grad);
    Weight = [C; W];
    %cor_MTL2 = eval_MTL_matthews(Y_te, X_te, W, C);

end

