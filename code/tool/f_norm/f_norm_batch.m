function xs_norm=f_norm_batch(res,theConf,theStat_all)
layerNum=length(theConf.convnet.targetLayers);
xs_norm(layerNum).x=[];
xs_norm(layerNum).valid=[];
for layer=1:layerNum
    theX=res(theConf.convnet.targetLayers(layer)).x;
    xs_norm(layer).valid=(theX>0);
    xs_norm(layer).x=(theX-theStat_all.avg_res(layer).x)./theStat_all.sqrtvar_layer(layer).x;
end
