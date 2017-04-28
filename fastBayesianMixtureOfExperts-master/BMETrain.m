%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Liefeng Bo, Cristian Sminchisescu                         %                                         
% Date: 01/12/2010                                                   %
%                                                                    % 
% Copyright (c) 2010  L. Bo, C. Sminchisescu - All rights reserved   %
%                                                                    %
% This software is free for non-commercial usage only. It must       %
% not be distributed without prior permission of the author.         %
% The author is not responsible for implications from the            %
% use of this software. You can run it at your own risk.             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function BME = BMETrain(BME, MTLTarget, trainTarget, testTarget)

%% Train BME

count = 1;
while (count <= BME.MaxIt)
        
    for i = 1:BME.NumExperts    
        BME = BMEExpertsTrain(MTLTarget, BME, i) ;
        BME = BMEGatingsTrain(BME, i) ;
        BME.Gatings.Outputs = exp(BME.Gatings.Input*BME.Gatings.Weights);
    end
    
    BME.Experts.Means = BMEExpertsMeans(BME.Experts.Input, BME.Experts.Weights, BME.Experts.Category_name, BME.Experts.Category_index);
    BME.Experts.Variances = BMEExpertsVariances(trainTarget, BME);
    BME.Gatings.Posteriors = BMEGatingsPosterior(trainTarget, BME);
    BME.LogLike(count,1) = BMELogLike(trainTarget, BME);
    %trainProb = sum(BME.Experts.Means.* BME.Gatings.Outputs, 2) ./ sum(BME.Gatings.Outputs, 2);
    %trainProb = sum(BME.Experts.Means.* BME.Gatings.Outputs, 2) ./ sum(BME.Gatings.Outputs, 2);
    %corr(trainProb > 0.5, trainTarget)
    if count == 1
        LogLikeChange = 10*BME.MinLogLikeChange*BME.LogLike(count);
    else
        LogLikeChange = BME.LogLike(count) - BME.LogLike(count-1);
    end
    TrainCategory_index = BME.Experts.Category_index;
    [BME.Test.TrainingMAE(count,:), BME.Test.TrainingPMAE(count,:)] = BMETest(BME.Experts.Input, TrainCategory_index, BME.Gatings.Input, trainTarget, BME);

    disp(['Current Iteration:               '  num2str(count)]);
    disp(['Current log likelihood:          '  num2str(BME.LogLike(count))]);
    if count > 1
        disp(['Previous log likelihood:         '  num2str(BME.LogLike(count-1))]);
        disp(['Log Likelihood Change:           '  num2str(LogLikeChange)]);
    end
    disp(['Best Training Error:             ' num2str(BME.Test.TrainingMAE(count,:))]);
    disp(['Most Probable Training Error:    ' num2str(BME.Test.TrainingPMAE(count,:))]);
    if nargin > 2
        TestECategory_index = BME.Test.ECategory_index;
        [BME.Test.TestMAE(count,:), BME.Test.TestPMAE(count,:)] = BMETest(BME.Test.EInput, TestECategory_index, BME.Test.GInput, testTarget, BME);
        disp(['Best Test Error:                 ' num2str(BME.Test.TestMAE(count,:))]);
        disp(['Most Probable Test Error:        ' num2str(BME.Test.TestPMAE(count,:))]);
    end
    disp('--------------------------------------------------------------'); 
    
    
    testMeans = BMEExpertsMeans(BME.Test.EInput, BME.Experts.Weights, BME.Experts.Category_name, BME.Test.ECategory_index);
    testGatingOutput = exp(BME.Test.GInput*BME.Gatings.Weights);
    sumTestGatingOutput = sum(testGatingOutput, 2);
    
    TestProb = sum(testMeans.* testGatingOutput, 2) ./ sumTestGatingOutput;
    
    
    corr_eval = corr(TestProb >= 0.5, testTarget);
    prec_rec(TestProb, testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    lgd = legend(num2str(corr_eval));
    title(lgd,'evaluation');
    disp('--------------------------------------------------------------');
    if ( abs(LogLikeChange) < BME.MinLogLikeChange*abs(BME.LogLike(count)))
        break;
    end
    count = count + 1;
end

if isfield(BME.Gatings,'InvH')
    BME.Gatings = rmfield(BME.Gatings,'InvH');
end
if isfield(BME.Gatings,'InvHH')
    BME.Gatings = rmfield(BME.Gatings,'InvHH');
end