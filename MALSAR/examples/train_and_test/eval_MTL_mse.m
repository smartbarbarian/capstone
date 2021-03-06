function mse = eval_MTL_mse (Y, X, W)
%% FUNCTION eval_MTL_mse
%   computation of mean squared error given a specific model.
%   the value is the lower the better.
%   
%% FORMULATION
%
%  multi-task mse = sum_t (mse(t) * N_t) / sum_t N_t
%
%  where 
%     mse(t) = sum((Yt_pred - Y{t})^2))/ N_t
%     Yt_pred = X{t} * W(:, t)
%     N_t     = length(Y{t})
%
%  which can be simplified as:
%  
%  multi-task mse = sum_t sum((Yt_pred - Y{t})^2)) / sum_t N_t
%  
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   percent: percentage of the splitting range (0, 1)
%
%% OUTPUT
%   X_sel: the split of X that has the specifid percent of samples 
%   Y_sel: the split of Y that has the specifid percent of samples 
%   X_res: the split of X that has the 1-percent of samples 
%   Y_res: the split of Y that has the 1-percent of samples 
%   selIdx: the selection index of for X_sel and Y_sel for each task
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on Jan 6, 2016.
%
    %Threshold = 0.8;
    task_num = length(X);
    %mse = 0;
    
    %total_sample = 0;
%     true_pos = 0;
%     pos_num = 0;
    y = [];
    y_true = [];
    for t = 1: task_num
          %y_pred = X{t} * W(:, t) + C(t);
          y_pred = glmval(W(:, t), X{t}, 'logit','constant','off');
          % y_pred = glmval(W(:, t), X{t}, 'logit');
          y_label = y_pred > 0.5;
          y = [y; y_label];
          y_true = [y_true; Y{t}];
        %y_pred = X{t} * W(:, t);
        %y_pred = glmval(W(:, t), X{t}, 'logit','constant','off');
        
%         prob_label = [y_pred Y{t}];
%         prob_label = sortrows(prob_label, - 1);%default ascending, unless you are using 2017b
%         
%         true_label_num = sum(Y{t} == 1);
%         threshold_num = Threshold*true_label_num;
%         now_true_num = 0;
%         for i = 1:length(Y{t})
%             if(prob_label(i, 2) == 1)
%                 now_true_num = now_true_num + 1;
%             end
%             if(now_true_num > threshold_num)
%                 pos_num = pos_num + i;
%                 true_pos = true_pos + now_true_num;
%                 break;
%             end
%         end
%         
        
        %mse = mse + (sum((y_pred - Y{t}).^2)/length(y_pred)) * length(y_pred);
        %mse = mse + sum((y_pred - Y{t}).^2); % length of y cancelled. 
        %total_sample = total_sample + length(y_pred);
    end
    y_true = y_true > 0;
    mse = corr(y, y_true); %actually is matthews
    %mse = true_pos / pos_num;
    %mse = mse/total_sample;
end
