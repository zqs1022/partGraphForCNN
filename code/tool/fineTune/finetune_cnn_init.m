function net=finetune_cnn_init(theConf,Name_batch)
%code for Computer Vision, Georgia Tech by James Hays

net=load(theConf.convnet.rawmodelname);
net=vl_simplenn_tidy(net);

net.layers=net.layers(1:end-2);
ly=struct('type','conv','weights',{{0.05.*randn(1,1,4096,2, 'single'), zeros(1,2,'single')}},'learningRate',[10,10],'stride',[1,1],'pad',[0,0,0,0],'name','fcE','dilate',1);
ly.opts={};
ly.precious=0;
net.layers{end+1}=ly;
net.layers{end+1}=struct('type','softmaxloss');
net.meta.classes.name={{Name_batch},{'neg'}};
net.meta.classes.description={{Name_batch},{'neg'}};

%We'll need to make some modifications to this network. First, the network
%accepts 

%This network is missing the dropout layers (because they're not needed at
%test time). It may be a good idea to reinsert dropout layers between the
%fully connected layers.
vl_simplenn_display(net, 'inputSize', [224 224 3 50])

% [copied from the project webpage]
% proj6_part2_cnn_init.m will start with net = load('imagenet-vgg-f.mat');
% and then edit the network rather than specifying the structure from
% scratch.

% You need to make the following edits to the network: The final two
% layers, fc8 and the softmax layer, should be removed and specified again
% using the same syntax seen in Part 1. The original fc8 had an input data
% depth of 4096 and an output data depth of 1000 (for 1000 ImageNet
% categories). We need the output depth to be 15, instead. The weights can
% be randomly initialized just like in Part 1.

% The dropout layers used to train VGG-F are missing from the pretrained
% model (probably because they're not used at test time). It's probably a
% good idea to add one or both of them back in between fc6 and fc7 and
% between fc7 and fc8.

