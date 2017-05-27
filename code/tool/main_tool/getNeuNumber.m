function neu=getNeuNumber(layer,theDepth,theConf,xh)
xw=xh;
num=theDepth.validNum;
if(num==0)
    neu=0;
    return;
end
validList=theDepth.nodeRank(1:num);
init=List2Geo_(validList,layer,xh,theConf);
SearchStride=round(theConf.convnet.SearchRange/2/init.Stride);
neu=0;
for i=1:num
    xhlen=min(init.xHW(1,i)+SearchStride,xh)-max(init.xHW(1,i)-SearchStride,1)+1;
    xwlen=min(init.xHW(2,i)+SearchStride,xw)-max(init.xHW(2,i)-SearchStride,1)+1;
    neu=neu+xhlen*xwlen;
end
end
