function coefficient = eval_MTL_matthews(Y, X, W, C)
%%Function eval_matthews
%computation matthews correlation of coefficient 
%which used for imbalance data
%the value is the higher the better
%   Detailed explanation goes here
    task_num = length(X);
    num = 0;
    for t = 1:task_num
        num = num + length(X{t});
    end
    y = zeros([num 1]);
    y_true = zeros([num 1]);
    start = 1;
    W = [C; W];
    for t = 1: task_num
          %y_pred = X{t} * W(:, t) + C(t);
          %y_label = y_pred >= 0; 
          
          %y_pred = glmval(W(:, t), X{t}, 'logit','constant','off');
          % y_pred = glmval(W(:, t), X{t}, 'logit');
          
          y_pred = glmval(W(:, t), X{t}, 'logit','constant','on');
          y_label = y_pred >= 0.5;
          
          y(start : start + length(X{t}) - 1) = y_label;
          y_true(start : start + length(X{t}) - 1) = Y{t};
          start = start + length(X{t});
    end
    y_true = y_true > 0;
    coefficient = corr(y, y_true); 

end

