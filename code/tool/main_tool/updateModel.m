function updateModel(label_set,parts_target,parts_target_corner,inference_target,invalidObjects,partID,step,Name_batch,theConf,IsPL)
parts_target=compressAOG(parts_target);
if(IsPL)
    parts_target_corner=compressAOG(parts_target_corner);
end
if(step==1)
    parts=parts_target;
    if(IsPL)
        parts_corner=reshape(parts_target_corner,[4,1]);
    end
    clear parts_target parts_target_corner;
    inference=inference_target;
else
    cateLabel_name_previous=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch,partID,step-1);
    load(cateLabel_name_previous,'parts','parts_corner','inference');
    poseID=label_set(end).poseID;
    parts(poseID)=parts_target;
    if(IsPL)
        parts_corner(:,poseID)=parts_target_corner;
    end
    clear parts_target parts_target_corner;
    for i=1:length(inference)
        if(~isempty(inference(i).objID))
            inference(i).poseGenDisL(poseID)=inference_target(i).poseGenDisL;
            inference(i).posePos(poseID)=inference_target(i).posePos;
            [~,idx]=max(inference(i).poseGenDisL);
            thePosition=inference(i).posePos(idx);
            if(poseID==idx)
                inference(i).map=inference_target(i).map;
            else
                inference(i).map=rmfield(thePosition,'flip');
            end
            inference(i).flip=thePosition.flip;
        end
    end
end
cateLabel_name=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch,partID,step);
if(IsPL)
    save(cateLabel_name,'parts','parts_corner','label_set','invalidObjects','inference');
else
    save(cateLabel_name,'parts','label_set','invalidObjects','inference');
end
end
