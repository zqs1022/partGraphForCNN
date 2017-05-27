function f=extractFC7(I_patch,net,tar_h,tar_w,scale_h,scale_w,IsFlip)
targetLayer=36; %35;

if(nargin<7)
    IsFlip=false;
end
[h,w,~]=size(I_patch);
if(IsFlip)
    I_patch=I_patch(:,end:-1:1,:);
    tar_w=w+1-tar_w;
end
minH=max(round(tar_h-scale_h/2),1);
minW=max(round(tar_w-scale_w/2),1);
maxH=min(round(tar_h+scale_h/2),h);
maxW=min(round(tar_w+scale_w/2),w);
scale_h=maxH-minH;
scale_w=maxW-minW;
tar_h=(maxH+minH)/2;
tar_w=(maxW+minW)/2;
tarSize=net.meta.normalization.imageSize(1:2);
if(size(I_patch,3)==1)
    I_patch=repmat(I_patch,[1,1,3]);
end
if(size(net.meta.normalization.averageImage,1)==1)
    I=repmat(net.meta.normalization.averageImage,[tarSize,1]);
else
    I=net.meta.normalization.averageImage;
end
I(minH:maxH,minW:maxW,:)=I_patch(minH:maxH,minW:maxW,:);
%I=single(imresize(I,tarSize,'bilinear'));
f_pre=[abs([tar_h,tar_w,minH,minW,maxH,maxW]./repmat(tarSize,[1,3])-0.5),[scale_h,scale_w]./tarSize.*1.5].*2;

if(size(net.meta.normalization.averageImage,1)==1)
    im_=single(I-repmat(net.meta.normalization.averageImage,[tarSize,1]));
else
    im_=single(I-net.meta.normalization.averageImage);
end
if(isa(net.layers{1}.weights{1},'gpuArray'))
    res=vl_simplenn(net,gpuArray(im_));
else
    res=vl_simplenn(net,im_);
end
f=gather(res(targetLayer).x);
f=double(reshape(f,[numel(f),1]));
f=[f_pre';f];
end
