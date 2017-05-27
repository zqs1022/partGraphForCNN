function step_showPartLocalization
% showPartLocalization('Exp1/n01443537','n01443537',1);
showPartLocalization('Exp1/n01503061','n01503061',2);
% showPartLocalization('Exp1/n01639765','n01639765',2);
% showPartLocalization('Exp1/n01662784','n01662784',2);
% showPartLocalization('Exp1/n01674464','n01674464',2);
% showPartLocalization('Exp1/n01882714','n01882714',2);
% showPartLocalization('Exp1/n01982650','n01982650',2);
% showPartLocalization('Exp1/n02084071','n02084071',2);
% showPartLocalization('Exp1/n02118333','n02118333',2);
% showPartLocalization('Exp1/n02121808','n02121808',2);
% showPartLocalization('Exp1/n02129165','n02129165',2);
% showPartLocalization('Exp1/n02129604','n02129604',2);
% showPartLocalization('Exp1/n02131653','n02131653',2);
% showPartLocalization('Exp1/n02324045','n02324045',2);
% showPartLocalization('Exp1/n02342885','n02342885',2);
% showPartLocalization('Exp1/n02355227','n02355227',2);
% showPartLocalization('Exp1/n02374451','n02374451',2);
% showPartLocalization('Exp1/n02391049','n02391049',2);
% showPartLocalization('Exp1/n02395003','n02395003',2);
% showPartLocalization('Exp1/n02398521','n02398521',2);
% showPartLocalization('Exp1/n02402425','n02402425',2);
% showPartLocalization('Exp1/n02411705','n02411705',2);
% showPartLocalization('Exp1/n02419796','n02419796',2);
% showPartLocalization('Exp1/n02437136','n02437136',2);
% showPartLocalization('Exp1/n02444819','n02444819',2);
% showPartLocalization('Exp1/n02454379','n02454379',2);
% showPartLocalization('Exp1/n02484322','n02484322',2);
% showPartLocalization('Exp1/n02503517','n02503517',2);
% showPartLocalization('Exp1/n02509815','n02509815',2);
% showPartLocalization('Exp1/n02510455','n02510455',2);
end


function showPartLocalization(Name_batch_model,Name_batch,partNum)
for partID=1:partNum
    showPartLocal(Name_batch_model,Name_batch,partID);
end
end


function showPartLocal(Name_batch_model,Name_batch,partID)
global conf;
theConf=conf;
iterNum=1;
while(true)
    cateLabel_name=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch_model,partID,iterNum+1);
    if(exist(cateLabel_name,'file')==2)
        iterNum=iterNum+1;
    else
        break;
    end
end
cateLabel_name=sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch_model,partID,iterNum);
load(cateLabel_name,'label_set','inference');
objset=readAnnotation(Name_batch,theConf);
va=[];
for par=1:length(inference)
    va=[va,inference(par).map.MEAN_GenDisL_est];
end
[~,order]=sort(va,'descend');
figroot=sprintf('./fig_out/fig_pl/%s_part%02d_',Name_batch,partID);
list=[];
for label=label_set
    list=[list,label.obj.ID];
end
c=0;
for par=1:10 %1:20:length(inference)
    tar=order(par);
    obj=objset(tar);
    if(sum(obj.ID==list)>0)
        continue;
    end
    c=c+1;
    [I_patch,~]=getI(obj,theConf,[224,224],inference(tar).flip);
    pI=uint8(I_patch);
    pMap=inference(tar).map.map;
    figure;imshow(pI);hold on;
    tar_h=inference(tar).map.position(1);
    tar_w=inference(tar).map.position(2);
    scale=inference(tar).map.scale;
    plot(tar_w+[-0.5,-0.5,0.5,0.5,-0.5].*scale(2),tar_h+[-0.5,0.5,0.5,-0.5,-0.5].*scale(1),'w-','LineWidth',2);
    set(gcf,'color','w');pause(0.2);
    saveas(gcf,sprintf('%s%02d.fig',figroot,c));
    figure;imagesc(pMap);axis off;set(gcf,'color','w');pause(0.2);
    saveas(gcf,sprintf('%s%02d_map.fig',figroot,c));
    close all;
end
end
