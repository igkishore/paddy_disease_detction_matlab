digitDatasetPath = fullfile('C:\Users\GOWTHAM KISHORE\Desktop\paddy by cnn');
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
inputSize = [64 220];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
img = readimage(imds,1);
size(img);
b=zeros(120,3);
 numberOfImages = length(imds.Files)
% for k = 1 : numberOfImages
%   % Get the input filename.  It already has the folder prepended so we don't need to worry about that.
%   inputFileName = imds.Files{k};
%   fprintf('Checking %s\n', inputFileName);
%   rgbImage = imread(inputFileName);
%   [rows, columns, numberOfColorChannels] = size(rgbImage);
%   if numberOfColorChannels == 3
%     % It's color so need to convert to gray scale.
%     grayImage = rgb2gray(rgbImage);
%     imshow(grayImage);
%     imwrite(grayImage,inputFileName);
%   else
%     % It's already gray scale
%     imshow(rgbImage);
%   end
%   axis('on', 'image');
%   drawnow;
%   pause(0.1); % Short delay so we can see the image a little bit.
% end


for i=1:120
    img = readimage(imds,1);
    b(i,1)=size(img,1);
    b(i,2)=size(img,2);
    b(i,3)=size(img,3);
end

labelCount = countEachLabel(imds);
img = readimage(imds,1);
disp(size(img))
numTrainFiles = 30;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    imageInputLayer([64 220 1])
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

options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',10,'Shuffle','every-epoch','ValidationData',imdsValidation,'ValidationFrequency',30,'Verbose',false,'Plots','training-progress');
 
labelIDs=[28 28 1];

classNames = ["Bacterial leaf blight" , "Brown spot" , "Leaf smut"];

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);

YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
cm = confusionchart(YTest,YPred);
mean_square_error=mean((-predicted_class_name).^2)

