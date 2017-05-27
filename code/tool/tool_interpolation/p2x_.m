function xHW=p2x_(pHW,layer,theConf,xh)
Stride=theConf.convnet.targetStride(layer);
centerStart=theConf.convnet.targetCenter(layer);
xHW=min(max(round((pHW-centerStart)./Stride+1),1),xh);
end
