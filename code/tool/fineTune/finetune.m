function [net, info] = finetune(theConf,objset,objset_neg,Name_batch,varargin)
%code for Computer Vision, Georgia Tech by James Hays
%based off the MNIST example from MatConvNet
run('vl_setupnn.m');

gpuNum=gpuDeviceCount;
%It might actually be problematic to run vl_setup, because VLFeat has a
%version of vl_argparse that conflicts with the matconvnet version. You
%shouldn't need VLFeat for this project.
% run(fullfile('vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));

%opts.expDir is where trained networks and plots are saved.
opts.expDir=[theConf.output.dir,Name_batch];
%delete([opts.expDir,'/net-epoch*']);

%opts.batchSize is the number of training images in each batch. You don't
%need to modify this.
opts.batchSize=50*gpuNum; %50; %256;

% opts.learningRate is a critical parameter that can dramatically affect
% whether training succeeds or fails. For most of the experiments in this
% project the default learning rate is safe.
opts.learningRate=0.0001; %0.00001; %0.0001;

% opts.numEpochs is the number of epochs. If you experiment with more
% complex networks you might need to increase this. Likewise if you add
% regularization that slows training.
opts.numEpochs=90;

% An example of learning rate decay as an alternative to the fixed learning
% rate used by default. This isn't necessary but can lead to better
% performance.
% opts.learningRate = logspace(-4, -5.5, 300) ;
% opts.numEpochs = numel(opts.learningRate) ;

%opts.continue controls whether to resume training from the furthest
%trained network found in opts.batchSize. If you want to modify something
%mid training (e.g. learning rate) this can be useful. You might also want
%to resume a network that hit the maximum number of epochs if you think
%further training can improve accuracy.
opts.continue=false;

%GPU support is off by default.
opts.gpus=1:gpuNum;

% This option lets you control how many of the layers are fine-tuned.
% opts.backPropDepth = 2; %just retrain the last real layer (1 is softmax)
% opts.backPropDepth = 9; %just retrain the fully connected layers
% opts.backPropDepth = +inf; %retrain all layers [default]

net=finetune_cnn_init(theConf,objset(1).name);
theSize=net.meta.normalization.imageSize(1:2);
imdb=finetune_setup_data(objset,objset_neg,theConf,theSize);

if(size(net.meta.normalization.averageImage,1)==1)
    targetAvgImg=repmat(net.meta.normalization.averageImage,[theSize,1,size(imdb.images.data,4)]);
else
    targetAvgImg=repmat(net.meta.normalization.averageImage,[1,1,1,size(imdb.images.data,4)]);
end
imdb.images.data=imdb.images.data-targetAvgImg;
clear targetAvgImg
[net,info]=cnn_train(net,imdb,@getBatch,opts,'val',find(imdb.images.set==1));
delete([opts.expDir,'/net-epoch*']);

end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
%getBatch is called by cnn_train.

%'imdb' is the image database.
%'batch' is the indices of the images chosen for this batch.

%'im' is the height x width x channels x num_images stack of images. If
%opts.batchSize is 50 and image size is 64x64 and grayscale, im will be
%64x64x1x50.
%'labels' indicates the ground truth category of each image.

%This function is where you should 'jitter' data.
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% Add jittering here before returning im
end