function parts=initializeParts(label,res,theConf)
layerNum=length(theConf.convnet.targetLayers);
parts(1).layer(layerNum).depth=[];
for layer=1:layerNum
    x=f(res,layer,theConf);
    [xh,xw,depthNum]=size(x);
    nodeRank=(1:xh*xw)';
    tmp=List2Geo_(nodeRank,layer,xh,theConf);
    pHW=x2p_(tmp.xHW,layer,theConf);
    DeltaHW=repmat(label.pHW_center,[1,xh*xw])-pHW;
    HoG=label.app(layer).HOG;
    LabelNum=1;
    depth=getStructure(nodeRank,DeltaHW,HoG,LabelNum);
    parts(1).layer(layer).depth=repmat(depth,[1,depthNum]);
end
end


function depth=getStructure(nodeRank,DeltaHW,HoG,LabelNum)
depth.nodeRank=nodeRank;
depth.DeltaHW=DeltaHW;
depth.HoG=HoG;
depth.LabelNum=LabelNum;
num=length(nodeRank);
depth.parent(num).DepID=[];
depth.parent(num).DeltaHW=[];
depth.validNum=num;
end
