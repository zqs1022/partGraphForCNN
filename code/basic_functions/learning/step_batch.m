function step_batch(Name_batch,partID,label_set,theConf)
thePara_center=@getLossPara_normal;
thePara_center_infer=[];
load([theConf.output.dir,Name_batch,'/all_statistics.mat'],'stat_all');
theStat_all=stat_all;clear stat_all;
load([theConf.output.dir,Name_batch,'/net_finetune.mat'],'net','info');
if(theConf.QA.IsUseGPU)
    net=vl_simplenn_move(net,'gpu');
end
file_genLoss=[theConf.output.dir,Name_batch,'/genLoss.mat'];
load(file_genLoss,'Loss');
invalidObjects=[];
labelNum=length(label_set);
labelID=size(labelNum,1);
for i=1:labelNum
    labelID(i)=label_set(i).poseID;
end
poseNum=max(labelID);
for step=1:poseNum
    targetLabelSet=label_set(labelID==step);
    IsPL=false;
    IsRough=true;
    target=getTargetLabelSetAndParts(targetLabelSet,Name_batch,partID,net,theConf,step,IsPL);
    parts_target=updateModel_pose(target.label_set,target.parts,net,theStat_all,Loss,theConf,target.IsNewPose,thePara_center);
    inference_target=step_inference(Name_batch,target.label_set,parts_target,theStat_all,Loss,net,IsRough,thePara_center_infer,theConf);
    updateModel(targetLabelSet,parts_target,[],inference_target,invalidObjects,partID,step,Name_batch,theConf,IsPL);
end
end
