function theBest=getGenDisL_layerBatch(x_layer,valid_layer,gen_batch,layer,part_batch,fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara)
maxPoolMemSize=250000;

depthNum=size(x_layer,3);
theBest=repmat(struct('best',[]),[1,depthNum]);
parentNum=[];
d_start=0;
while(true)
    d_batch=0;
    validNum=0;
    while(true)
        d_batch=d_batch+1;
        d=d_start+d_batch;
        vList=part_batch(d).nodeRank;
        len=length(vList);
        if(len~=part_batch(d).validNum)
            error('errors in function getGenDisL_layerBatch.');
        end
        validNum=validNum+len;
        if(isempty(parentNum)&&(len>0))
            parentNum=size(part_batch(d).parent(1).DepID,2);
        end
        if((validNum>=maxPoolMemSize)||(d==depthNum))
            break;
        end
    end
    batch_list=d_start+1:d_start+d_batch;
    theBest(batch_list)=getGenDisL_batch(validNum,parentNum,x_layer(:,:,batch_list),valid_layer(:,:,batch_list),gen_batch(batch_list),layer,part_batch(batch_list),fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara);
    if(~theConf.QA.IsUseGPU)
        fprintf('Processed getGenDisL_batch for %d slices in layer %d\n',d_batch,layer)
    end
    d_start=d_start+d_batch;
    if(d_start==depthNum)
        break;
    end
end


function theBest=getGenDisL_batch(validNum,parentNum,x_layer,valid_layer,gen_batch,layer,part_batch,fset,pHW_avg,parent_record,theConf,label_pHW_center,thePara)
depthNum=size(x_layer,3);
HoGDim=size(fset,1);
tmp=repmat(struct('DepID',[],'DeltaHW',[]),[1,validNum]);
part_layer=struct('nodeRank',zeros(validNum,1),'DeltaHW',zeros(2,validNum),'HoG',zeros(HoGDim,validNum),'parent',tmp,'validNum',validNum);
gen_layer=struct('SUM',zeros(validNum,1),'local',zeros(validNum,1),'geo',zeros(validNum,1));
depth_layer=zeros(validNum,1);
parent_layer=struct('pHW',zeros(2,parentNum,validNum),'valid',zeros(parentNum,validNum));
c=0;
len_record=zeros(depthNum,1);
for d=1:depthNum
    len=length(part_batch(d).nodeRank);
    if(len==0)
        continue;
    end
    len_record(d)=len;
    tarlist=c+1:c+len;
    part_layer.nodeRank(tarlist)=part_batch(d).nodeRank;
    part_layer.DeltaHW(:,tarlist)=part_batch(d).DeltaHW;
    part_layer.HoG(:,tarlist)=part_batch(d).HoG;
    part_layer.parent(tarlist)=part_batch(d).parent;
    gen_layer.SUM(tarlist)=gen_batch(d).SUM;
    gen_layer.local(tarlist)=gen_batch(d).local;
    gen_layer.geo(tarlist)=gen_batch(d).geo;
    depth_layer(tarlist)=d;
    tmp=getParentAdvice(part_batch(d),parent_record);
    parent_layer.pHW(:,:,tarlist)=tmp.pHW;
    parent_layer.valid(:,tarlist)=tmp.valid;
    c=c+len;
end
clear part_batch gen_batch parent_record
best=getGenDisL_core(x_layer,valid_layer,part_layer,depth_layer,layer,gen_layer,fset,pHW_avg,label_pHW_center,parent_layer,theConf,thePara);
tmpBest=struct('pHW_part',[],'xHW',[],'local',[],'geo',[],'app',[],'gen_local',[],'gen_geo',[],'parent',[],'super_loc',[],'GenL',[],'DisL',[],'reli',[],'GenDisL',[]);
if(isempty(label_pHW_center))
    tmpBest=rmfield(tmpBest,{'super_loc','DisL','GenDisL'});
end
theBest=repmat(struct('best',[]),[1,depthNum]);
c=0;
for d=1:depthNum
    len=len_record(d);
    if(len==0)
        continue;
    end
    list=c+1:c+len;
    theBest(d).best=tmpBest;
    theBest(d).best.pHW_part=best.pHW_part(:,list);
    theBest(d).best.xHW=best.xHW(:,list);
    theBest(d).best.local=best.local(list);
    theBest(d).best.geo=best.geo(list);
    theBest(d).best.app=best.app(list);
    theBest(d).best.gen_local=best.gen_local(list);
    theBest(d).best.gen_geo=best.gen_geo(list);
    theBest(d).best.parent=best.parent(list);
    theBest(d).best.GenL=best.GenL(list);
    theBest(d).best.reli=best.reli(list);
    if(~isempty(label_pHW_center))    
        theBest(d).best.super_loc=best.super_loc(list);
        theBest(d).best.DisL=best.DisL(list);
        theBest(d).best.GenDisL=best.GenDisL(list);
    end
    c=c+len;
end
