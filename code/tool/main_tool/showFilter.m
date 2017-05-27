function showFilter(xHWList,layerList,I,showPercent)
transparentRate=0.7;

global conf;
theConf=conf;
color=colormap('jet');
close gcf
layerNum=length(theConf.convnet.targetLayers);
color=color(round(linspace(1,size(color,1),layerNum)),:);

num=length(layerList);
numList=round(linspace(1,num,round(num*showPercent)));
figure;
imshow(I);
hold on;
pause(0.2);
for i=numList
    layer=layerList(i);
    diff=theConf.convnet.targetScale(layer)/2;
    pHW=x2p_(xHWList(:,i),layer,theConf);
    %plot(pHW(2)+[-1,-1,1,1,-1].*diff,pHW(1)+[-1,1,1,-1,-1].*diff,'Color',color(layer,:));
    drawCircle(pHW(1),pHW(2),diff,color(layer,:));
end
hold off
pause(0.2);

figure;
imshow(I);
hold on;
pause(0.2);
map=zeros(size(I));
for i=numList
    layer=layerList(i);
    pHW=x2p_(xHWList(:,i),layer,theConf);
    %theSize=3*(1.5^layer);
    %plot(pHW(2),pHW(1),'o','MarkerFaceColor',color(layer,:),'MarkerEdgeColor',color(layer,:),'MarkerSize',theSize);
    diff=theConf.convnet.targetScale(layer)/2;
    map=drawSolidCircle(pHW(1),pHW(2),diff,color(layer,:),map);
end
map=map.*255;
I=double(I).*transparentRate+map.*(1-transparentRate);
I=uint8(I);
imshow(I);
hold off
pause(0.2);


function drawCircle(h,w,r,col)
tmp=linspace(0,2*pi,51);
x=w+sin(tmp).*r;
y=h+cos(tmp).*r;
plot(x,y,'Color',col);


function map=drawSolidCircle(y,x,r,col,map)
[h,w,d]=size(map);
xList=repmat(1:w,[h,1]);
yList=repmat((1:h)',[1,w]);
validList=find((xList-x).^2+(yList-y).^2<=r^2);
map=reshape(map,[h*w,d]);
map(validList,:)=repmat(col,[length(validList),1]);
map=reshape(map,[h,w,d]);
