function parts_target=updateModel_pose(label_set,parts,net,theStat_all,Loss,theConf,IsNewPose,thePara)
%%% the layerNum=theConf.convnet.smallLayerNum for parts_corner
layerNum=length(parts.layer);
pHW_avg=get_pHW_avg(label_set,label_set(end).poseID);
depthAll=0;
L=repmat(struct('depth',[]),[1,layerNum]);
for layer=layerNum:-1:1
    depthNum=size(Loss(layer).Gen.SUM,2);
    L(layer).depth=repmat(struct('value',[],'xHW',[],'app',[],'DeltaHW',[],'nodeRank',[],'N',[]),[1,depthNum]);
    depthAll=depthAll+depthNum;
end
for id=1:length(label_set)
    label=label_set(id);
    label_pHW_center=label.pHW_center;
    obj=label.obj;
    [res,~,~]=getObjFeature(obj,theConf,net,label.flip);
    clear parSet;
    parSet=repmat(struct('best',[]),[1,depthAll]);
    parent_record=[];
    c=0;
    [parts_pose,gen]=getPartsAndLoss_withInvalid(Loss,parts); %%%%%% not function getPartsAndLoss_withInvalid for a comprehensive learning
    for layer=layerNum:-1:1
        v=f_norm(res,layer,theConf,theStat_all);
        fset=label.app(layer).HOG;
        depthNum=size(v.x,3);
        %tic
        parSet(c+1:c+depthNum)=getGenDisL_layerBatch(v.x,v.valid,gen(layer).depth,layer,parts_pose.layer(layer).depth,fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara);
        %parSet(c+1:c+depthNum)=getGenDisL_depthBatch(v.x,v.valid,gen(layer).depth,layer,parts_pose.layer(layer).depth,fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara);
        %toc
        parent_record=getParentLayerRecord(parSet(c+1:c+depthNum),parts_pose.layer(layer).depth,layer,theConf);
        c=c+depthNum;
    end
    c=0;
    for layer=layerNum:-1:1
        depthNum=length(L(layer).depth);
        for d=1:depthNum
            c=c+1;
            part=parts.layer(layer).depth(d);
            if(isempty(parSet(c).best)||isempty(parSet(c).best.xHW))
                continue;
            end
            the_xHW=parSet(c).best.xHW;
            the_pHW=x2p_(the_xHW,layer,theConf);
            DeltaHW=repmat(label_pHW_center,[1,size(the_pHW,2)])-the_pHW;
            xh=sqrt(length(Loss(layer).Gen.SUM(:,d)));
            xHWList=HW2List_(the_xHW,xh);
            if(isempty(L(layer).depth(d).N))
                L(layer).depth(d).value=parSet(c).best.GenDisL;
                L(layer).depth(d).xHW=the_xHW;
                L(layer).depth(d).app=label.app(layer).HOG(:,xHWList);
                L(layer).depth(d).DeltaHW=DeltaHW;
                L(layer).depth(d).nodeRank=part.nodeRank;
                L(layer).depth(d).N=1;
            else
                L(layer).depth(d).value=L(layer).depth(d).value+parSet(c).best.GenDisL;
                L(layer).depth(d).xHW=L(layer).depth(d).xHW+the_xHW;
                if(theConf.QA.IsUseHoG), L(layer).depth(d).app=L(layer).depth(d).app+label.app(layer).HOG(:,xHWList); end
                L(layer).depth(d).DeltaHW=L(layer).depth(d).DeltaHW+DeltaHW;
                L(layer).depth(d).N=L(layer).depth(d).N+1;
            end
        end
    end
    fprintf('Processing the No.%d annotation.\n',id)
end
clear parts
parts_target.layer=repmat(struct('depth',[]),[1,layerNum]);
[weight,~,~]=thePara();
for layer=layerNum:-1:1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    depthNum=length(L(layer).depth);
    parts_target.layer(layer).depth=repmat(struct('validNum',[],'nodeRank',[],'DeltaHW',[],'HoG',[],'LabelNum',[]),[1,depthNum]);
    xh=sqrt(size(Loss(layer).Gen.SUM,1));
    bestList=[];
    valueRecord=repmat(struct('value',[]),[1,depthNum]);
    for d=1:depthNum
        theN=L(layer).depth(d).N;
        if(~isempty(theN))
            value=L(layer).depth(d).value./theN;
            [value,order]=sort(value,'descend');
            xHW=round(L(layer).depth(d).xHW(:,order)./theN);
            DeltaHW=L(layer).depth(d).DeltaHW(:,order)./theN;
            HoG=L(layer).depth(d).app(:,order)./theN;
            newRank=HW2List_(xHW,xh);
            [newRank,order]=unique(newRank,'stable');
            parts_target.layer(layer).depth(d).nodeRank=newRank;
            parts_target.layer(layer).depth(d).LabelNum=theN;
            parts_target.layer(layer).depth(d).DeltaHW=DeltaHW(:,order);
            parts_target.layer(layer).depth(d).HoG=HoG(:,order);
            value=value(order);
            bestList(end+1:end+length(value))=value;
            valueRecord(d).value=value;
            %%% update parent
            if(layer==layerNum)
                tmpParent.DepID=[];
                tmpParent.DeltaHW=[];
                TheParent=repmat(tmpParent,[1,length(newRank)]);
            else
                xh_parent=sqrt(size(Loss(layer+1).Gen.SUM,1));
                TheParent=updateParent(parts_target.layer(layer+1).depth,newRank,layer,theConf,xh,xh_parent,weight.parentNum);
            end
            parts_target.layer(layer).depth(d).parent=TheParent;
        end
    end
    %%% update validNum
    [bestList,~]=sort(bestList,'descend');
    para=regressionDistri(bestList);
    topRank=min(ceil(weight.nodeNum_range/para.lambda),length(bestList));
    if(isfield(weight,'nodeNum_max'))
        topRank=min(weight.nodeNum_max,topRank);
    end
    va=bestList(topRank);
    for d=1:depthNum
        parts_target.layer(layer).depth(d).validNum=sum(valueRecord(d).value>=va);
    end
end
end


function para=regressionDistri(y)
%y=alpha.*exp(-(lambda.*x).^index)+beta
index=0.5;
tail=0.1;
topRank=0.1;
y=y(1:ceil(length(y)*topRank));
para.beta=y(end);
y=y(y>(y(1)-y(end))*tail+y(end));
len=length(y);
x=[(0:len-1)'.^index,ones(len,1)];
y_tar=reshape(log(y-para.beta),[len,1]);
tmp=((x'*x)\(x'))*y_tar;
para.lambda=(-tmp(1))^(1/index);
para.alpha=exp(tmp(2));

% figure;hold on;
% plot(1:length(y_tar),y_tar);
% weight=getLossPara_highReli();
% topRank=min(ceil(weight.nodeNum_range/para.lambda),length(y_tar));
% plot([topRank,topRank],[min(y_tar),max(y_tar)]);
% pause;
end
