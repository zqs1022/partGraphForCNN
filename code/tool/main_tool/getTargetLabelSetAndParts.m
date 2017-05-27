function target=getTargetLabelSetAndParts(label_set,Name_batch,partID,net,theConf,step,IsPL)
if(nargin<7)
    IsPL=false;
end
label=label_set(end);
target=initializeTarget(label,theConf,net,IsPL);
% if(step==1)
%     target=initializeTarget(label,theConf,net,IsPL);
% else
%     cateLabel_name_previous=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch,partID,step-1);
%     load(cateLabel_name_previous,'parts','parts_corner');
%     if(label.poseID<=length(parts))
%         target.parts=uncompressAOG(parts(label.poseID));
%         if(IsPL)
%             target.parts_corner=uncompressAOG(parts_corner(:,label.poseID));
%         end
%         target.IsNewPose=false;
%         target.poseID=label.poseID;
%     else
%         target=initializeTarget(label,theConf,net,IsPL);
%     end
% end
labelNum=length(label_set);
list=zeros(1,labelNum);
c=0;
for i=1:labelNum
    if(target.poseID==label_set(i).poseID)
        c=c+1;
        list(c)=i;
    end
end
list=list(1:c);
target.label_set=label_set(list);
end
