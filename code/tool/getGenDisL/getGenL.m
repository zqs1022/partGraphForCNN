function best=getGenL(x,valid,layer,theConf,weight)
[h,w,depthNum]=size(x);
validList=(1:h*w)';
init=List2Geo_(validList,layer,h,theConf);
SearchStride=round(theConf.convnet.SearchRange/2/init.Stride);
value=getValue(x,valid,init,0,0,weight);
best=value;
best.geo=repmat(best.geo,[h*w,depthNum]);
best.xHW=repmat(init.xHW,[1,depthNum]);
for dh=-SearchStride:SearchStride
    for dw=-SearchStride:SearchStride
        if((dh==0)&&(dw==0))
            continue;
        end
        [value,xHW]=getValue(x,valid,init,dh,dw,weight);
        list=find(value.SUM>best.SUM);
        if(~isempty(list))
            best.SUM(list)=value.SUM(list);
            best.local(list)=value.local(list);
            best.geo(list)=value.geo;
            xHW=repmat(xHW,[1,depthNum]);
            best.xHW(:,list)=xHW(:,list);
        end
    end
end
%size(reshape((ones(h*w,1)*(0:depthNum-1)).*(h*w),[h*w*depthNum,1]))
best.xHW=HW2List_(best.xHW,h)+reshape((ones(h*w,1)*(0:depthNum-1)).*(h*w),[h*w*depthNum,1]);
best.weight=weight;
end


function [value,xHW]=getValue(x,valid,init,dh,dw,weight)
minValue=-1000000;
inValidX=-3;

[h,w,depthNum]=size(x);
value.geo=getLogGauss_([dh;dw],init.Scale/init.Stride);
value.local=repmat(minValue,[h*w,depthNum]);
xHW=init.xHW+repmat([dh;dw],[1,h*w]);
list=find((min(xHW,[],1)>0).*(xHW(1,:)<=h).*(xHW(2,:)<=w)>0);
x=reshape(x,[h*w,depthNum]);
valid=reshape(valid,[h*w,depthNum]);
x(valid==0)=inValidX;
value.local(list,:)=x(HW2List_(xHW(:,list),h),:);
value.SUM=value.local.*weight.local+value.geo.*weight.geo;
end
