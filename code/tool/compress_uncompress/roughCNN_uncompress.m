function res=roughCNN_uncompress(res_c,theConf)
num=length(res_c);
res=repmat(struct('x',[],'dzdx',[],'dzdw',[],'aux',[],'stats',[],'time',0,'backwardTime',0),[1,num]);
for i=theConf.convnet.targetLayers
    theSize=res_c(i).size;
    x=(single(res_c(i).x).*repmat(single(res_c(i).rangeX),[theSize(1),theSize(2),1]))./65535.0;
    x=x+repmat(single(res_c(i).minX),[theSize(1),theSize(2),1]);
    res(i).x=reshape(x,theSize);
end
end
