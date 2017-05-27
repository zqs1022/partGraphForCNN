function record=getParentLayerRecord(parSet,theDepth,layer,theConf)
depthNum=length(parSet);
record(depthNum).valid=[];
record(depthNum).pHW=[];
for d=1:depthNum
    validNum=theDepth(d).validNum;
    if(~isempty(parSet(d).best))
        the_xHW=parSet(d).best.xHW(:,1:validNum);
        the_local=parSet(d).best.local(1:validNum);
        record(d).valid=double(the_local>0);
        record(d).pHW=x2p_(the_xHW,layer,theConf);
    end
end
end
