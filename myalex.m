alex = alexnet; % loading alexnet CNN
layers = alex.Layers; % assigning the layers of Alexnet to a variable
%% Modify the network to use five categories
%%daisy dandelion  rose  sunflower  tulip
layers(23) = fullyConnectedLayer(5); % modifying the fully connected layer for five
layers(25) = classificationLayer
clear alex; % free up some memory
%% Set up our training data
% myImages is a folder of 2352 images with five sub-folders,
% average images of approximately 500 for each category
allImages = imageDatastore('myImages', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Spliting the imagedatastore into 2 partition training imamges and testing
% images in a ration of 90:10
[trainingImages, testImages] = splitEachLabel(allImages, 0.9, 'randomize');
clear allImages;
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 1, 'MiniBatchSize', 64);
myAlexNet = trainNetwork(trainingImages, layers, opts);
save myAlexNet; % Saving the CNN
%% Measure network accuracy
predictedLabels = classify(myAlexNet, testImages); % comparing the test images with training images
accuracy = mean(predictedLabels == testImages.Labels);% accuracy calculation
save myAlexNetaccuracy
[filename, pathname] = uigetfile('*.*', 'Pick a Image');
filename=strcat(pathname,filename);
I=imread(filename);
imshow(I)
[label,scores] = classify( myAlexNet,I);
imshow(I)
net=myAlexNet;
classNames = net.Layers(25).ClassNames;
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");
