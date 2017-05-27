function label=label_part(obj,theConf,theNet,theStat_all,IsFlip)
MinArea=200;
[res,I_patch,~]=getObjFeature(obj,theConf,theNet,IsFlip);
figure;
imshow(I_patch);
hold on;
[h,w,~]=size(I_patch);
while(true)
    waitforbuttonpress;
    point1=get(gca,'CurrentPoint');
    rbbox;
    point2=get(gca,'CurrentPoint');
    point1=point1(1,1:2);
    point2=point2(1,1:2);
    MinH=min(point1(2),point2(2));
    MaxH=max(point1(2),point2(2));
    MinW=min(point1(1),point2(1));
    MaxW=max(point1(1),point2(1));
    if((MinH>h)||(MaxH<0)||(MinW>w)||(MaxW<0))
        continue;
    else
        if((MaxH-MinH)*(MaxW-MinW)<MinArea)
            continue;
        else
            break;
        end
    end
end

label.pHW_center=[(MinH+MaxH)/2;(MinW+MaxW)/2];
label.pHW_scale=[MaxH-MinH;MaxW-MinW];
label.I=I_patch;
label.obj=obj;
layerNum=length(theConf.convnet.targetLayers);
label.app(layerNum).HOG=[];
for layer=1:layerNum
    v=f_norm(res,layer,theConf,theStat_all);
    %label.app(layer).HOG=getHoG_(I_patch,layer,theConf,size(v.x,1));
    label.app(layer).HOG=x2normApp(v.x,theConf);
end
plot([MinW,MaxW,MaxW,MinW,MinW],[MinH,MinH,MaxH,MaxH,MinH]);
pause(0.1);
