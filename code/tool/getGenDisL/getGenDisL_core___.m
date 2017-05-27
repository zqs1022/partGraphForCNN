function best=getGenDisL_core(x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,theConf)
minGPUUnit=50000;

theConf.QA.IsUseGPU=theConf.QA.IsUseGPU&&(length(part.nodeRank)>=minGPUUnit);
[weight,inValidX,minValue]=getLossPara();
[xh,xw,~]=size(x);
init=List2Geo_(part.nodeRank,layer,xh,theConf);
if(theConf.QA.IsUseGPU)
    best=getGenDisL_core_GPU(init,x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,xh,xw,weight,inValidX,minValue,theConf);
else
    best=getGenDisL_core_GPU(init,x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,xh,xw,weight,inValidX,minValue,theConf);
end
end


function best=getGenDisL_core_GPU(init,x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,xh,xw,weight,inValidX,minValue,theConf)
if(theConf.QA.IsUseGPU)
    init.xHW=gpuArray(init.xHW);
    x=gpuArray(x);
    valid=gpuArray(valid);
    part.nodeRank=gpuArray(part.nodeRank);
    part.DeltaHW=gpuArray(part.DeltaHW);
    depth=gpuArray(depth);
    layer=gpuArray(layer);
    gen.local=gpuArray(gen.local);
    gen.geo=gpuArray(gen.geo);
    gen.SUM=gpuArray(gen.SUM);
    fset=gpuArray(fset);
    pHW_avg.scale=gpuArray(pHW_avg.scale);
    pHW_avg.center=gpuArray(pHW_avg.center);
    label_pHW_center=gpuArray(label_pHW_center);
    parent.pHW=gpuArray(parent.pHW);
    parent.valid=gpuArray(parent.valid);
end
[value,pHW_part]=getValue(x,valid,gen,init,0,0,part,layer,depth,fset,pHW_avg,label_pHW_center,parent,theConf,weight,minValue,inValidX,xh,xw);
best=value;
best.pHW_part=round(pHW_part);
SearchStride=round(theConf.convnet.SearchRange/2/init.Stride);
for dh=-SearchStride:SearchStride
    for dw=-SearchStride:SearchStride
        if((dh==0)&&(dw==0))
            continue;
        end
        [value,pHW_part]=getValue(x,valid,gen,init,dh,dw,part,layer,depth,fset,pHW_avg,label_pHW_center,parent,theConf,weight,minValue,inValidX,xh,xw);
        list=find(value.GenL>best.GenL);
        best.pHW_part(:,list)=pHW_part(:,list);
        best.xHW(:,list)=value.xHW(:,list);
        best.local(list)=value.local(list);
        best.geo(list)=value.geo(list);
        best.app(list)=value.app(list);
        best.gen_local(list)=value.gen_local(list);
        best.gen_geo(list)=value.gen_geo(list);
        best.gen_SUM(list)=value.gen_SUM(list);
        best.parent(list)=value.parent(list);
        best.super_loc(list)=value.super_loc(list);
        best.GenL(list)=value.GenL(list);
        best.DisL(list)=value.DisL(list);
        best.reli(list)=value.reli(list);
        best.GenDisL(list)=value.GenDisL(list);
    end
end
if(theConf.QA.IsUseGPU)
    best.pHW_part=gather(best.pHW_part);
    best.xHW=gather(best.xHW);
    best.local=gather(best.local);
    best.geo=gather(best.geo);
    best.app=gather(best.app);
    best.gen_local=gather(best.gen_local);
    best.gen_geo=gather(best.gen_geo);
    best.gen_SUM=gather(best.gen_SUM);
    best.parent=gather(best.parent);
    best.super_loc=gather(best.super_loc);
    best.GenL=gather(best.GenL);
    best.DisL=gather(best.DisL);
    best.reli=gather(best.reli);
    best.GenDisL=gather(best.GenDisL);
end
if(isempty(label_pHW_center))
    best=rmfield(best,{'super_loc','DisL','GenDisL'});
end
end


function list=getLayIndex(xHW,xh,xw,depth)
list=(xHW(1,:)+(xHW(2,:)-1).*xh)'+reshape((depth-1),[numel(depth),1]).*(xh*xw);
end

function [value,pHW_part]=getValue(x,valid,gen,init,dh,dw,part,layer,depth,fset,pHW_avg,label_pHW_center,parent,theConf,weight,minValue,inValidX,xh,xw)
num=size(init.xHW,2);
pHW_part=zeros(2,num);
tmp=repmat(minValue,[num,1]);
value.parent=zeros(num,1);
if(theConf.QA.IsUseGPU)
    tmp=gpuArray(tmp);
    pHW_part=gpuArray(pHW_part);
    value.parent=gpuArray(value.parent);
    inValidX=gpuArray(inValidX);
    dh=gpuArray(dh);
    dw=gpuArray(dw);
    xh=gpuArray(xh);
    xw=gpuArray(xw);
    weight.local=gpuArray(weight.local);
    weight.geo=gpuArray(weight.geo);
    weight.app=gpuArray(weight.app);
    weight.reli=gpuArray(weight.reli);
    weight.gen=gpuArray(weight.gen);
    weight.parent=gpuArray(weight.parent);
    weight.super_loc=gpuArray(weight.super_loc);
end
value.local=tmp;
value.geo=tmp;
value.app=tmp;
value.gen_local=tmp;
value.gen_geo=tmp;
value.gen_SUM=tmp;
value.xHW=init.xHW+repmat([dh;dw],[1,num]);
value.super_loc=tmp; %zeros(num,1); %%%%%%%%%%%
value.reli=tmp;
value.GenL=tmp;
value.DisL=tmp;
value.GenDisL=tmp;

list=find((min(value.xHW,[],1)>0).*(value.xHW(1,:)<=xh).*(value.xHW(2,:)<=xw)>0);
listlen=length(list);
tmpList=getLayIndex(value.xHW(:,list),xh,xw,depth(list));
value.local(list)=x(tmpList);
tmpValid=valid(tmpList);
value.local(list(tmpValid==0))=inValidX;
delta=sqrt(init.Stride^2+pHW_avg.scale.^2);
value.geo(list)=getLogGauss_([dh;dw],delta/init.Stride);
value.app(list)=-(sum((part.HoG(:,list)-fset(:,list)).^2,1));
pHW_init=x2p_(init.xHW(:,list),layer,theConf);
mu_reli=repmat(pHW_avg.center,[1,listlen])-pHW_init;
value.reli(list)=getLogGauss_(mu_reli,norm(pHW_avg.scale));
pHW=x2p_(value.xHW(:,list),layer,theConf);
pHW_est=pHW+part.DeltaHW(:,list);
pHW_part(:,list)=pHW_est;
value.gen_local(list)=gen.local(list);
value.gen_geo(list)=gen.geo(list);
value.gen_SUM(list)=gen.SUM(list);
if(~isempty(parent.valid))
    parentNum=size(parent.pHW,2);
    mu_parent=repmat(reshape(pHW,[2,1,listlen]),[1,parentNum,1])-parent.pHW(:,:,list);
    expGain=reshape(getExpGain_(mu_parent,delta./2),[parentNum,listlen]);
    value.parent(list)=mean(expGain.*parent.valid(:,list),1)';
end
if(~isempty(label_pHW_center))
    mu_loc=repmat(label_pHW_center,[1,listlen])-pHW_est;
    value.super_loc(list)=getLogGauss_(mu_loc,pHW_avg.scale./2); %%%%%%%%%%%%
end
GenLoss=value.local.*weight.local+value.geo.*weight.geo+value.app.*weight.app+value.gen_SUM.*weight.gen+value.parent.*weight.parent;
value.GenL(list)=GenLoss(list);
DisLoss=value.super_loc.*weight.super_loc;
value.DisL(list)=DisLoss(list);
value.GenDisL(list)=value.GenL(list)+value.DisL(list)+value.reli(list).*weight.reli;

% disp([value.local(list)'.*weight.local;
% value.geo(list)'.*weight.geo;
% value.app(list)'.*weight.app;
% value.reli(list)'.*weight.reli;
% value.gen_SUM(list)'.*weight.gen;
% value.parent(list)'.*weight.parent;
% value.GenL(list)';
% value.DisL(list)';
% value.GenDisL(list)']);
% 'abc'
% pause
end
