function [appNum,latNum,neuNum]=getNodeNum(parts,xh,conf)
parts=uncompressAOG(parts);
layerNum=length(parts(1).layer);
appNum=length(parts);
latNum=0;
neuNum=0;
for poseID=1:appNum
    for layer=1:layerNum
        theDepth=parts(poseID).layer(layer).depth;
        for d=1:length(theDepth)
            latNum=latNum+theDepth(d).validNum;
            neuNum=neuNum+getNeuNumber(layer,theDepth(d),conf,xh(layer));
        end
    end
end
end
