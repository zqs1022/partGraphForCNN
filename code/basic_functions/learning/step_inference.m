function [inference_target,xHW_est]=step_inference(Name_batch,label_set,parts,theStat_all,Loss,net,IsRough,thePara_center,theConf)
IsOnlyForTruth=false; %%%%%%%%%%%%%%%%%

if(nargin<7)
    IsRough=false;
    disp('You are using accurate inference, which will cost more time. You can choose the rough inference to save time by setting IsRough=true.');
    pause;
end
if((nargin<8)||(isempty(thePara_center)))
    thePara_center=@getLossPara_normal;
end
objset=readAnnotation(Name_batch,theConf);
objNum=length(objset);
validObj=validObjList(theConf,Name_batch,objNum,IsOnlyForTruth);
batchSize=theConf.QA.parpoolSize*3;
if(IsRough)
    filename_roughCNN=[theConf.output.dir,Name_batch,'/roughCNN.mat'];
    load(filename_roughCNN,'roughCNN');
end
poseID=label_set(end).poseID;
pHW_avg=get_pHW_avg(label_set,poseID);
%%% produce the generative loss map and subtract invalid parts
layerNum=length(theConf.convnet.targetLayers);
[parts,gen]=getPartsAndLoss_withoutInvalid(Loss,parts);
parSet=repmat(struct('layerNum',layerNum,'part',parts,'gen',gen,'conf',theConf,'pHW_avg',pHW_avg,'xs_norm',[],'I',[],'xs_norm_flip',[],'I_flip',[],'objID',[],'classResponse',[]),[1,objNum]);
inference_target=repmat(struct('map',[],'objID',[],'flip',[],'poseGenDisL',[],'posePos',[],'classResponse',[]),[1,objNum]);
xHW_est=repmat(struct('xHW',[]),[layerNum,objNum]);
vObjList=find(validObj==1);
for idx=1:batchSize:length(vObjList)
    objList=vObjList(idx:min(length(vObjList),idx+batchSize-1));
    for par=objList
        obj=objset(par);
        [I_patch,~]=getI(obj,theConf,net.meta.normalization.imageSize(1:2),false);
        parSet(par).I=I_patch;
        if(IsRough)
            res=roughCNN_uncompress(roughCNN.res{par},theConf);
            res_flip=roughCNN_uncompress(roughCNN.res_flip{par},theConf);
        else
            [res,~,~]=getObjFeature(obj,theConf,net,false);
            [res_flip,~,~]=getObjFeature(obj,theConf,net,true);
        end
        parSet(par).xs_norm=f_norm_batch(res,theConf,theStat_all);
        parSet(par).xs_norm_flip=f_norm_batch(res_flip,theConf,theStat_all);
        parSet(par).I_flip=I_patch(:,end:-1:1,:);
        parSet(par).objID=obj.ID;
        %parSet(par).classResponse=(res(end).x(:,:,2)+res_flip(end).x(:,:,2))./2;
    end
    parfor par=min(objList):max(objList)
    %for par=objList
        if(sum(par==objList)==0)
            continue;
        end
        layNum=parSet(par).layerNum;
        xs_norm=parSet(par).xs_norm;
        xs_norm_flip=parSet(par).xs_norm_flip;
        pPart=parSet(par).part;
        pConf=parSet(par).conf;
        p_pHW_avg=parSet(par).pHW_avg;
        pI=parSet(par).I;
        pI_flip=parSet(par).I_flip;
        pGen=parSet(par).gen;
        parSet(par).xs_norm=[];
        parSet(par).xs_norm_flip=[];
        parSet(par).part=[];
        parSet(par).I=[];
        parSet(par).I_flip=[];
        parSet(par).pHW_avg=[];
        parSet(par).conf=[];
        [map,xHW_obj]=getMap(pI,xs_norm,layNum,pGen,pPart,p_pHW_avg,pConf,thePara_center);
        [map_flip,xHW_obj_flip]=getMap(pI_flip,xs_norm_flip,layNum,pGen,pPart,p_pHW_avg,pConf,thePara_center);
        if(map.MEAN_GenDisL_est>map_flip.MEAN_GenDisL_est)
            inference_target(par).map=map;
            inference_target(par).flip=false;
            xHW_est(:,par)=xHW_obj;
        else
            inference_target(par).map=map_flip;
            inference_target(par).flip=true;
            xHW_est(:,par)=xHW_obj_flip;
        end
        tmp=rmfield(inference_target(par).map,{'map','maxValue'});
        tmp.flip=inference_target(par).flip;
        inference_target(par).posePos=tmp;
        inference_target(par).poseGenDisL=inference_target(par).map.MEAN_GenDisL_est;
        inference_target(par).objID=parSet(par).objID;
        inference_target(par).classResponse=parSet(par).classResponse;
        parSet(par).gen=[];
        parSet(par).classResponse=[];
        disp(par);
    end
end
end


function [map,xHW_obj]=getMap(pI,xs_norm,layNum,pGen,pPart,p_pHW_avg,pConf,thePara)
mapSize=[size(pI,1),size(pI,2)];
parent_record=[];
best_record=repmat(struct('depth',[]),[1,layNum]);
xHW_obj=repmat(struct('xHW',[]),[layNum,1]);
for lay=layNum:-1:1
    [xh,~,~]=size(xs_norm(lay).x);
    %fset=getHoG_(pI,lay,pConf,xh);
    fset=x2normApp(xs_norm(lay).x,pConf);
    theDepth=getGenDisL_layerBatch(xs_norm(lay).x,xs_norm(lay).valid,pGen(lay).depth,lay,pPart.layer(lay).depth,fset,p_pHW_avg,parent_record,pConf,[],thePara);
    best_record(lay).depth=theDepth;
    parent_record=getParentLayerRecord(theDepth,pPart.layer(lay).depth,lay,pConf);
    depthNum=length(theDepth);
    theXHW=zeros((xh^2)*depthNum,1);
    c=0;
    for d=1:depthNum
        best=theDepth(d).best;
        if(isempty(best))
            continue;
        end
        tmp=size(best.xHW,2);
        theXHW(c+1:c+tmp)=HW2List_(best.xHW,xh)+(d-1)*(xh^2);
        c=c+tmp;
    end
    xHW_obj(lay).xHW=theXHW(1:c);
end
[map,~]=produceMap(best_record,pConf,p_pHW_avg,mapSize,thePara);
end
