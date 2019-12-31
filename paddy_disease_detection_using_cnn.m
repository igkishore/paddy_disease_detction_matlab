DatasetPath = fullfile('C:\Users\GOWTHAM KISHORE\Documents\MATLAB\paddy dataset');
imds = imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
inputSize = [64 220];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
labelCount = countEachLabel(imds);
[imdsValidation,imdsTrain] = splitEachLabel(imds,0.8,'randomize');
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    imageInputLayer([64 220 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];
options  = trainingOptions('sgdm','MaxEpochs',1000,'ValidationData',imdsValidation,'ValidationFrequency',3000,'Verbose',false, 'Plots','training-progress');
labelIDs=[28 28 1];
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = (sum(YPred == YValidation)/numel(YValidation))*100