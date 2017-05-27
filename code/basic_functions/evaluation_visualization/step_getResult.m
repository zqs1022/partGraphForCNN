function [result,normDist,LOCRate,dist]=step_getResult(Name_batch,partID,theConf)
minIOU=0.5;
ImgWidth=224;

Name_batch_target=Name_batch;
testingTruthFile=sprintf('truth_part%02d.mat',partID);
load(sprintf('%s%s/%s',theConf.output.dir,Name_batch,testingTruthFile),'truth');
objNum=length(truth);
validImgNum=0;
IsHaveScale=false;
for i=1:objNum
    if(~isempty(truth(i).obj))
        validImgNum=validImgNum+1;
    end
    if(isfield(truth(i),'pHW_scale'))
        IsHaveScale=(IsHaveScale||(~isempty(truth(i).pHW_scale)));
    end
end
for i=1:objNum
    if(~IsHaveScale)
        truth(i).pHW_scale=repmat([1;1],[1,size(truth(i).pHW_center,2)]);
    end
end
stepNum=0;
while(true)
    cateLabel_name=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch_target,partID,stepNum+1);
    if(exist(cateLabel_name,'file')==0)
        break;
    end
    stepNum=stepNum+1;
end

cateLabel_name=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch_target,partID,stepNum);
clear inference
load(cateLabel_name,'inference','label_set');
fprintf('label number %d\n',length(label_set))
if(length(inference)~=objNum)
    error('Errors in function step_showCurve.');
end
IOU=zeros(validImgNum,1);
LOC=zeros(validImgNum,1);
dist=zeros(validImgNum,1);
c=0;
for i=1:objNum
    if(~isempty(truth(i).obj))
        c=c+1;
        if(inference(i).flip)
            tarP=inference(i).map.position;
            tarP(2)=ImgWidth+1-tarP(2);
        else
            tarP=inference(i).map.position;
        end
        box=getBox(tarP,inference(i).map.scale);
        candidateNum=size(truth(i).pHW_center,2);
        dist(c)=99999999;
        for j=1:candidateNum
            box_t=getBox(truth(i).pHW_center(:,j),truth(i).pHW_scale(:,j));
            tmp1=min(box.wmax,box_t.wmax)-max(box.wmin,box_t.wmin)+1;
            tmp2=min(box.hmax,box_t.hmax)-max(box.hmin,box_t.hmin)+1;
            if((tmp1>0)&&(tmp2>0))
                intersection=tmp1*tmp2;
            else
                intersection=0;
            end
            union=box.area+box_t.area-intersection;
            iou=intersection/union;
            loc=(tarP(1)<=box_t.hmax)*(tarP(1)>=box_t.hmin)*(tarP(2)<=box_t.wmax)*(tarP(2)>=box_t.wmin);
            IOU(c)=max(IOU(c),iou);
            LOC(c)=max(LOC(c),loc);
            theDist=norm(truth(i).pHW_center(:,j)-tarP);
            dist(c)=min(dist(c),theDist);
        end
    end
end
dist=dist./(ImgWidth*sqrt(2));
result=mean(IOU>=minIOU)*100;
normDist=mean(dist);
LOCRate=mean(LOC)*100;

% dist=sort(dist);dist=dist(1:ceil(0.2*length(dist)));
% normDist=mean(dist);
end


function box=getBox(p,scale)
box.wmin=round(p(2)-scale(2)/2);
box.wmax=round(p(2)+scale(2)/2);
box.hmin=round(p(1)-scale(1)/2);
box.hmax=round(p(1)+scale(1)/2);
box.area=(box.wmax-box.wmin+1)*(box.hmax-box.hmin+1);
end
