%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Liefeng Bo, Cristian Sminchisescu                         %                                         
% Date: 01/12/2010                                                   %
%                                                                    % 
% Copyright (c) 2010  Liefeng Bo - All rights reserved               %
%                                                                    %
% This software is free for non-commercial usage only. It must       %
% not be distributed without prior permission of the author.         %
% The author is not responsible for implications from the            %
% use of this software. You can run it at your own risk.             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ExpertsMeans = BMEExpertsMeans(BME)
%% Compute the mean of experts
category_names = BME.Experts.Category_name;
category_indexs = BME.Experts.Category_index;
weights = BME.Experts.Weights;
inputs = BME.Experts.MTLinput;
col = length(category_names);% expert num
row = size(BME.Experts.Input, 1);
ExpertsMeans = zeros([row, col]);
for i = 1 : col
    input = inputs{i};
    category_name = category_names{i};
    category_index = category_indexs{i};
    weight = weights{i};
    
    task_num = length(category_name);
    for j = 1 : task_num
        %y_pred = glmval(W(:, t), X{t}, 'logit','constant','on');
        %output = input{j}*weight{j};
        
        output = glmval(weight(:, j), input{j}, 'logit','constant','on');
        isSelected = strcmp(category_index, category_name(j));
        ExpertsMeans(i, isSelected) = output;
    end
end

% if length(size(BME.Experts.Weights)) == 2
%     ExpertsMeans = Input*BME.Experts.Weights;
% else
%     for i = 1:size(BME.Experts.Weights,3)
%         ExpertsMeans(:,:,i) = Input*BME.Experts.Weights(:,:,i);
%     end
% end