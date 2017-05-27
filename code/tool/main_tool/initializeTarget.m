function target=initializeTarget(label,theConf,net,IsPL)
[res,~,~]=getObjFeature(label.obj,theConf,net,label.flip);
target.parts=initializeParts(label,res,theConf);
if(IsPL)
    tmp=target.parts;
    tmp.layer=tmp.layer(1:theConf.convnet.smallLayerNum);
    target.parts_corner=repmat(tmp,[4,1]);
    clear tmp
end
target.IsNewPose=true;
target.poseID=label.poseID;
end
