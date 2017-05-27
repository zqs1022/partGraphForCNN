function normDist=step_showCurve(Name_batch,modelDir,modelDir2,partID)
if(~isempty(modelDir2))
    modelDir2=[modelDir2,'/'];
end
global conf;
theConf=conf;

ImgWidth=224;
load(sprintf('%s%s/truth_part%02d.mat',theConf.output.dir,Name_batch,partID),'truth');
objNum=length(truth);
validImgNum=0;
for i=1:objNum
    if(~isempty(truth(i).obj))
        validImgNum=validImgNum+1;
    end
end
scale=zeros(validImgNum,1);
c=0;
for i=1:objNum
    if(~isempty(truth(i).obj))
        c=c+1;
        scale(c)=mean(sqrt(sum((truth(i).pHW_scale).^2,1)),2);
    end
end
stepNum=0;
while(true)
    cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,stepNum+1);
    if(exist(cateLabel_name,'file')==0)
        break;
    end
    stepNum=stepNum+1;
end
dist=zeros(validImgNum,stepNum);
pose=zeros(validImgNum,stepNum);
for step=1:stepNum
    cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,step);
    clear inference
    load(cateLabel_name,'inference');
    if(length(inference)~=objNum)
        error('Errors in function step_showCurve.');
    end
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
            candidateNum=size(truth(i).pHW_center,2);
            dist(c,step)=min(sqrt(sum((repmat(tarP,[1,candidateNum])-truth(i).pHW_center).^2,1)),[],2);
            [~,tarPose]=max(inference(i).poseGenDisL);
            pose(c,step)=tarPose;
        end
    end
end
figure;
hold on;
tmp=dist./repmat(scale,[1,stepNum]);
curve_0=mean(tmp,1);
curve_1=mean((tmp<=0.5),1);
curve_2=mean((tmp<=1.0),1);
%plot(1:stepNum,curve_0,'-k');
%plot(1:stepNum,curve_1,'-r');
%plot(1:stepNum,curve_2,'-b');
normDist=mean(dist./(224*sqrt(2)),1);
plot(1:stepNum,normDist,'-b');
pause
for step=1:stepNum
    figure;
    hist(pose(:,step));
end
end
