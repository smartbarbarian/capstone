function [ Weight ] = Grad_MTL( X, Y,  param, Grad)
%UNTITLED2 Summary of this function goes here
    %% balance
    task_num = length(X);
    X_ba = cell([1 task_num]);
    Y_ba = cell([1 task_num]);
    Grad_ba = cell([1 task_num]);
    for t = 1: task_num
        X_majority = X{t}(Y{t}(:,1) == 0, :);
        Y_majority = Y{t}(Y{t}(:,1) == 0, :);
        Grad_majority = Grad{t}(Y{t}(:,1) == 0, :);
        [X_minority, Y_minority] = ADASYN([X{t} Grad{t}], Y{t}, [], [], [], true);
        if isempty(X_minority)
            X_ba{t} = X{t};
            Grad_ba{t} = Grad{t};
            Y_ba{t} = double(Y{t});
        else    
            X_ba{t} = [X_minority(:, 1 : end - 1); X_majority];
            Grad_ba{t} = [X_minority(:, end); Grad_majority];
            Y_ba{t} = double([Y_minority; Y_majority]);
        end    
        
        Y_ba{t}(Y_ba{t} == 0) = - 1;
    end
    %% train
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l21%%%%%%%%%
    %rng('default');     % reset random generator. Available from Matlab 2011.
    opts.init = 0;      % guess start point from data. 
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-5;   % tolerance. 
    opts.maxIter = 1000; % maximum iteration number of optimization.
    [W, C, funcVal] = Logistic_L21(X_ba, Y_ba, param, opts, Grad_ba);
    Weight = [C; W];
    %cor_MTL2 = eval_MTL_matthews(Y_te, X_te, W, C);

end

