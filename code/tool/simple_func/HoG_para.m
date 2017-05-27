function HOG=HoG_para(layer,theConf)
HOG.VoteOriNum=6;
HOG.CellSize=theConf.convnet.targetStride(layer);
scale=theConf.convnet.targetScale(layer);
HOG.CellNum=round(scale/HOG.CellSize);
HOG.CellStart=theConf.convnet.targetCenter(layer);
end

