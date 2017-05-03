function [ Weight , Param] = Grad_MTL( X, Y, holdoutInput, holdoutTarget,  param, Grad, preWeight)
%UNTITLED2 Summary of this function goes here
    %% convert 0,1 to -1, 1
    %     for t = 1: size(X,2)
    %         Y{t} = double(Y{t});
    %         Y{t}(Y{t} == 0) = -1;
    %     end    
    %% train
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%
    %rng('default');     % reset random generator. Available from Matlab 2011.
    if isempty(preWeight)
        opts.init = 0;      % guess start point from data. 
    else
        opts.init = 1;
        opts.C0 = preWeight(1,:);
        opts.W0 = preWeight(2:end,:);
    end    
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-5;   % tolerance. 
    opts.maxIter = 1000; % maximum iteration number of optimization.
    
    training_percent = 0.5;
    [X_tr, Y_tr, X_te, Y_te, Grad_tr, Grad_te] = mtSplitPerc(X, Y, training_percent, Grad);
    
    [W, C, funcVal] = Logistic_Trace(X_tr, Y_tr, param, opts, Grad_tr);
    corr_cur = eval_MTL_matthews(holdoutTarget, holdoutInput, W, C);
    
    [W, C, funcVal] = Logistic_Trace(X_tr, Y_tr, param * 10, opts, Grad_tr);
    corr_big_next = eval_MTL_matthews(holdoutTarget, holdoutInput, W, C);
    
    param_big = param;
    corr_big = corr_cur;
    while corr_big_next > corr_big
        corr_big = corr_big_next;
        param_big = param_big * 10;
        [W, C, funcVal] = Logistic_Trace(X_tr, Y_tr, param_big * 10, opts, Grad_tr);
        corr_big_next = eval_MTL_matthews(holdoutTarget, holdoutInput, W, C);
    end
    
    [W, C, funcVal] = Logistic_Trace(X_tr, Y_tr, param / 10.0, opts, Grad_tr);
    corr_small_next = eval_MTL_matthews(holdoutTarget, holdoutInput, W, C);
    
    param_small = param;
    corr_small = corr_cur;
    while corr_small_next > corr_small
        corr_small = corr_small_next;
        param_small = param_small / 10;
        [W, C, funcVal] = Logistic_Trace(X_tr, Y_tr, param_small / 10, opts, Grad_tr);
        corr_small_next = eval_MTL_matthews(holdoutTarget, holdoutInput, W, C);
    end
    if corr_big > corr_small
        param = param_big;
    else
        param = param_small;
    end    
    %[W, C, funcVal] = Logistic_L21(X, Y, param, opts, Grad);
    [W, C, funcVal] = Logistic_Trace(X, Y, param, opts, Grad);
    
    Weight = [C; W];
    Param = param;
    %cor_MTL2 = eval_MTL_matthews(Y_te, X_te, W, C);

end

