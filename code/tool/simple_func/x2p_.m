function pHW=x2p_(xHW,layer,theConf)
Stride=theConf.convnet.targetStride(layer);
centerStart=theConf.convnet.targetCenter(layer);
pHW=centerStart+(xHW-1).*Stride;
end
