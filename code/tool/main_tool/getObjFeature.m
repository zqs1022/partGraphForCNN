function [res,I_patch,I]=getObjFeature(obj,theConf,theNet,IsFlip)
[I_patch,I]=getI(obj,theConf,theNet.meta.normalization.imageSize(1:2),IsFlip);
if(size(theNet.meta.normalization.averageImage,1)==1)
    im_=I_patch-repmat(theNet.meta.normalization.averageImage,[theNet.meta.normalization.imageSize(1:2),1]);
else
    im_=I_patch-theNet.meta.normalization.averageImage;
end
if(isa(theNet.layers{1}.weights{1},'gpuArray'))
    res=vl_simplenn(theNet,gpuArray(im_));
    for i=1:length(res)
        res(i).x=gather(res(i).x);
    end
else
    res=vl_simplenn(theNet,im_);
end
I_patch=uint8(I_patch);
end
