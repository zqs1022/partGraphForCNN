function heatMap=getHeatMap(xHWList,layerList,mapSize,theConf)
layerNum=length(theConf.convnet.targetScale);
heatMap=repmat(struct('map',zeros(mapSize)),[1,layerNum]);
num=length(layerList);
for i=1:num
    layer=layerList(i);
    pHW=round(x2p_(xHWList(:,i),layer,theConf));
    heatMap(layer).map=heatMap(layer).map+getGaussian(mapSize,pHW);
end
end


function template=getGaussian(mapSize,pHW)
range=40;
delta=15;
template=zeros(mapSize);
tmp_h=max(round(pHW(1)-range/2),1):min(round(pHW(1)+range/2),mapSize(1));
tmp_w=max(round(pHW(2)-range/2),1):min(round(pHW(2)+range/2),mapSize(2));
tmp=repmat((tmp_h'-pHW(1)).^2,[1,length(tmp_w)])+repmat((tmp_w-pHW(2)).^2.,[length(tmp_h),1]);
template(tmp_h,tmp_w)=exp(-tmp./(2*(delta^2)));
end
