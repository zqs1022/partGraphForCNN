function conf=configurations

conf.data.devkitdir='../data/devkit';
conf.data.minArea=50^2;

conf.data.catedir='../data/VOC_part/';
conf.data.imgdir='../data/VOC_part/VOCdevkit/VOC2010/JPEGImages/';
conf.data.imgdir_neg='../data/neg/';
conf.data.readCode='./data_input/data_input_VOC/';

conf.output.dir='./mat/';

convnet.codedir='../matconvnet/matlab/';
convnet.rawmodelname='../matconvnet/imagenet-vgg-verydeep-16.mat';
convnet.convLayers=[1,3,6,8,11,13,15,18,20,22,25,27,29];
convnet.validLayers=[5,6,7,8,9,10,11,12,13];
conf.convnet=getConvNetPara(convnet);
conf.convnet.SearchRange=224*0.3;

try
    gpuDevice();
    conf.QA.IsUseGPU=true;
    fprintf('GPU valid.\n')
catch
    conf.QA.IsUseGPU=false;
    fprintf('GPU invalid.\n')
end
p=gcp;
conf.QA.parpoolSize=p.NumWorkers;
end


function convnet=getConvNetPara(convnet)
net=load(convnet.rawmodelname);
len=length(convnet.convLayers);
convnet.lastLayer=length(net.layers)-1;
convnet.targetLayers=1+convnet.convLayers;
convnet.targetScale=zeros(1,len);
convnet.targetStride=zeros(1,len);
convnet.targetCenter=zeros(1,len);
for i=1:len
    tarLay=convnet.convLayers(i);
    layer=net.layers{tarLay};
    if((~strcmp(layer.type,'conv'))||(var(layer.pad)>0)||(layer.size(1)~=layer.size(2))||(layer.stride(1)~=layer.stride(2)))
        error('Errors in function getConvNetPara.');
    end
    pad=layer.pad(1);
    scale=layer.size(1);
    stride=layer.stride(1);
    if(i==1)
        convnet.targetStride(i)=stride;
        convnet.targetScale(i)=scale;
        convnet.targetCenter(i)=(1+scale-pad*2)/2;
    else
        IsPool=false;
        poolStride=0;
        poolSize=0;
        poolPad=0;
        for j=convnet.convLayers(i-1)+1:tarLay-1
            if(strcmp(net.layers{j}.type,'pool'))
                IsPool=true;
                poolSize=net.layers{j}.pool(1);
                poolStride=net.layers{j}.stride(1);
                poolPad=net.layers{j}.pad(1);
            end
        end
        convnet.targetStride(i)=(1+IsPool*(poolStride-1))*stride*convnet.targetStride(i-1);
        convnet.targetScale(i)=convnet.targetScale(i-1)+IsPool*(poolSize-1)*convnet.targetStride(i-1)+convnet.targetStride(i)*(scale-1);
        if(IsPool)
            convnet.targetCenter(i)=(scale-pad*2-1)*poolStride*convnet.targetStride(i-1)/2+(convnet.targetCenter(i-1)+convnet.targetStride(i-1)*(poolSize-2*poolPad-1)/2);
        else
            convnet.targetCenter(i)=(scale-pad*2-1)*convnet.targetStride(i-1)/2+convnet.targetCenter(i-1);
        end
    end
end
convnet.targetLayers=convnet.targetLayers(convnet.validLayers);
convnet.targetScale=convnet.targetScale(convnet.validLayers);
convnet.targetStride=convnet.targetStride(convnet.validLayers);
convnet.targetCenter=convnet.targetCenter(convnet.validLayers);
convnet=rmfield(convnet,{'convLayers','validLayers'});
end
