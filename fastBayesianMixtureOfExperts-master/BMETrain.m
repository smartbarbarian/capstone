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

function BME = BMETrain(BME, MTLTarget, trainTarget, testTarget, holdout_MTL_target)

%% Train BME

count = 1;
while (count <= BME.MaxIt)
        
    for i = 1:BME.NumExperts    
        BME = BMEExpertsTrain(MTLTarget, holdout_MTL_target, BME, i) ;
    end
    
    testMeans = BMEExpertsMeans(BME.Test.EInput, BME.Experts.Weights, BME.Experts.Category_name, BME.Test.ECategory_index);
    BME.Experts.Means = BMEExpertsMeans(BME.Experts.Input, BME.Experts.Weights, BME.Experts.Category_name, BME.Experts.Category_index);
    
    %%
    testGatingOutput = exp(BME.Test.GInput*BME.Gatings.Weights);
    sumTestGatingOutput = sum(testGatingOutput, 2);

    TestProb = sum(testMeans.* testGatingOutput, 2) ./ sumTestGatingOutput;

    BME.Test.TestProb = [BME.Test.TestProb, TestProb];
    test_corr_eval = corr(TestProb >= 0.5, testTarget);
    disp(['Current test_corr_eval:               '  num2str(test_corr_eval)]);
    %%
    preCorrEval = test_corr_eval;
    preGatingsWeights = BME.Gatings.Weights;
%     while true
        for i = 1:BME.NumExperts    
            BME = BMEGatingsTrain(BME, i) ;
        end
        %% hold out validation


        testGatingOutput = exp(BME.Test.GInput*BME.Gatings.Weights);
        sumTestGatingOutput = sum(testGatingOutput, 2);

        TestProb = sum(testMeans.* testGatingOutput, 2) ./ sumTestGatingOutput;

        BME.Test.TestProb = [BME.Test.TestProb, TestProb];
        test_corr_eval = corr(TestProb >= 0.5, testTarget);
        disp(['Current test_corr_eval:               '  num2str(test_corr_eval)]);
        
        if preCorrEval > test_corr_eval
            BME.Gatings.Weights = preGatingsWeights;
%             break;
        end     
            BME.Gatings.Outputs = exp(BME.Gatings.Input*BME.Gatings.Weights);
            BME.Gatings.Posteriors = BMEGatingsPosterior(trainTarget, BME);
%             preCorrEval = test_corr_eval;
%             preGatingsWeights = BME.Gatings.Weights;
%         end    
%     end

%         newPosteriors = BMEGatingsPosterior(trainTarget, BME);
%         newVariances = BMEExpertsVariances(trainTarget, BME);
%         BME.Gatings.Posteriors = newPosteriors;
%         BME.Experts.Variances = newVariances;
%         LogLike = BMELogLike(trainTarget, BME);
%         LogLikeChange = LogLike - preLogLike;
%         if ( abs(LogLikeChange) < BME.MinLogLikeChange*abs(preLogLike))
%             break;
%         end
%         preLogLike = LogLike;
    
    
    
    
%     newPosteriors = BMEGatingsPosterior(trainTarget, BME);
%     newVariances = BMEExpertsVariances(trainTarget, BME);
%     BME.Gatings.Posteriors = newPosteriors;
%     BME.Experts.Variances = newVariances;
%     
%     preLogLike = BMELogLike(trainTarget, BME);
    
       
    
    
    
    
    BME.LogLike(count,1) = BMELogLike(trainTarget, BME);
    %trainProb = sum(BME.Experts.Means.* BME.Gatings.Outputs, 2) ./ sum(BME.Gatings.Outputs, 2);
    %trainProb = sum(BME.Experts.Means.* BME.Gatings.Outputs, 2) ./ sum(BME.Gatings.Outputs, 2);
    %corr(trainProb > 0.5, trainTarget)
    
    %trainResult = (BME.Experts.Means > 0.5)
    %corrResult = corr(trainResult, trainTarget)
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
    
    
    testGatingOutput = exp(BME.Test.GInput*BME.Gatings.Weights);
    sumTestGatingOutput = sum(testGatingOutput, 2);
    
    TestProb = sum(testMeans.* testGatingOutput, 2) ./ sumTestGatingOutput;
    
    BME.Test.TestProb = [BME.Test.TestProb, TestProb];
    corr_eval = corr(TestProb >= 0.5, testTarget);
    prec_rec(TestProb, testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
    lgd = legend(num2str(corr_eval));
    title(lgd,'evaluation');
    disp('--------------------------------------------------------------');
%     if ( abs(LogLikeChange) < BME.MinLogLikeChange*abs(BME.LogLike(count)))
    if (abs(LogLikeChange) < 1e-8)
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