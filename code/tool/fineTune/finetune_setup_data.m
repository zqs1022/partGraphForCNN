function imdb=finetune_setup_data(objset,objset_neg,theConf,image_size)
%code for Computer Vision, Georgia Tech by James Hays

%This path is assumed to contain 'test' and 'train' which each contain 15
%subdirectories. The train folder has 100 samples of each category and the
%test has an arbitrary amount of each category. This is the exact data and
%train/test split used in Project 4.

num_pos=length(objset);
num_neg=length(objset_neg);
total_images=num_pos*2+num_neg;
imdb.images.data=zeros(image_size(1),image_size(2),3,total_images,'single');
imdb.images.labels=zeros(1,total_images,'single');
imdb.images.set=zeros(1,total_images,'uint8');
imdb.images.labels(1,1:num_pos*2)=1;
imdb.images.labels(1,num_pos*2+1:end)=2;
for i=1:num_pos
    tar=i*2-1;
    IsFlip=false;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    imdb.images.data(:,:,:,tar)=I_patch;
    tar=i*2;
    IsFlip=true;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    imdb.images.data(:,:,:,tar)=I_patch;
end
theConf_neg=theConf;
theConf_neg.data.imgdir=theConf_neg.data.imgdir_neg;
for i=1:num_neg
    tar=num_pos*2+i;
    IsFlip=false;
    [I_patch,~]=getI(objset_neg(i),theConf_neg,image_size,IsFlip);
    imdb.images.data(:,:,:,tar)=I_patch;
end
imdb.images.set(1:end)=1;
