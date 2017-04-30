train = readtable('trainData.csv','ReadRowNames',true);
test = readtable('testData.csv','ReadRowNames',true);
BME = MOEMTL(train(:,2:end), train(:,1), 4, test(:,2:end), test(:,1));
%trainData, trainLabel, categoricalColNum, testData, testLabel
trainData = train(:,2:end);
trainLabel = train(:,1);
categoricalColNum = 4;
testData = test(:,2:end);
testLabel = test(:,1);
corr_eval = corr(TestProb >= 0.5, testTarget);
prec_rec(TestProb, testTarget, 'plotPR', 1, 'plotROC', 0, 'holdFigure', 1, 'plotBaseline', 0);
lgd = legend(num2str(corr_eval));