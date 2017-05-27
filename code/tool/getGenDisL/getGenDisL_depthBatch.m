function theBest=getGenDisL_depthBatch(x_layer,valid_layer,gen_batch,layer,part_batch,fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara)
depthNum=size(x_layer,3);
parSet=repmat(struct('x',[],'valid',[],'gen',[],'fset',[],'label_pHW_center',label_pHW_center,'pHW_avg',pHW_avg,'layer',layer,'d',[],'part',[],'conf',theConf,'parent',[]),[1,depthNum]);
theBest=repmat(struct('best',[]),[1,depthNum]);
for d=1:depthNum
    part=part_batch(d);
    validList=part.nodeRank;
    TheX=squeeze(x_layer(:,:,d));
    TheValid=squeeze(valid_layer(:,:,d));
    parent=getParentAdvice(part,parent_record);
    parSet(d).x=TheX;
    parSet(d).valid=TheValid;
    parSet(d).gen=gen_batch(d);
    parSet(d).fset=fset(:,validList);
    parSet(d).d=d;
    parSet(d).part=part;
    parSet(d).parent=parent;
end
parfor par=1:depthNum
    if(~isempty(parSet(par).part.nodeRank))
        tic
    	theBest(par).best=getGenDisL_depth(parSet(par).x,parSet(par).valid,parSet(par).gen,parSet(par).layer,parSet(par).part,parSet(par).fset,parSet(par).pHW_avg,parSet(par).parent,parSet(par).conf,parSet(par).label_pHW_center,thePara);
        toc
    end
    parSet(par).x=[];
    parSet(par).gen=[];
    parSet(par).fset=[];
    parSet(par).part=[];
    parSet(par).conf=[];
    fprintf('getGenDisL_depthBatch: layer %02d  depth %02d\n',parSet(par).layer,par)
end


function best=getGenDisL_depth(x,valid,gen,layer,part,fset,pHW_avg,parent,theConf,label_pHW_center,thePara)
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
best=getGenDisL_core(x,valid,part,depth,layer,gen,fset,pHW_avg,label_pHW_center,parent,theConf,thePara);
