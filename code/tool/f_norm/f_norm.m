function v=f_norm(res,layer,theConf,theStat_all)
x=gather(res(theConf.convnet.targetLayers(layer)).x);
v.x=(x-theStat_all.avg_res(layer).x)./theStat_all.sqrtvar_layer(layer).x;
v.valid=(x>0);
end
