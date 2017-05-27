function best=getGenDisL_depthBatch(x,valid,gen,layer,part,fset,pHW_avg,parent,theConf,label_pHW_center)
theNum=length(part.nodeRank);
if((isempty(label_pHW_center))&&(theNum~=part.validNum))
    error('errors in function getGenDisL, using invalid nodes for inference');
end
if((length(part.parent)~=theNum)||(size(part.DeltaHW,2)~=theNum)||(size(part.HoG,2)~=theNum))
    error('errors in function getGenDisL, in the term of "part".');
end
if(size(parent.valid,2)>theNum)
    error('errors in function getGenDisL, in the term of "parent".');
end
if(size(fset,2)~=theNum)
    error('errors in function getGenDisL, in the term of "fset".');
end
if((size(x,3)>1)||(size(valid,3)>1))
    error('errors in function getGenDisL, inputing more than one depth of "x" and/or "valid".');
end
depth=ones(theNum,1);
best=getGenDisL_core(x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,theConf);
