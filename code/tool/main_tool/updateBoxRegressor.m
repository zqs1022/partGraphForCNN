function updateBoxRegressor(label_set,Name_batch,partID,net,stepList,antList,theConf,minSize)
root_out=[theConf.output.dir,Name_batch,'/boxRegression/'];
mkdir(root_out);
num=length(label_set);
for i=1:num
    I_patch=label_set(i).I;
    label=label_set(i);
    tar_h=label.pHW_center(1);
    tar_w=label.pHW_center(2);
    scale_h=max(label.pHW_scale(1),minSize);
    scale_w=max(label.pHW_scale(2),minSize);
    f_pos=getPosF(I_patch,tar_h,tar_w,scale_h,scale_w,net);
    f_neg=getNegF(I_patch,tar_h,tar_w,scale_h,scale_w,net);
    if(i==1)
        [dimNum,negNum]=size(f_neg);
        [~,posNum]=size(f_pos);
        fea_pos=zeros(num*posNum,dimNum);
        fea_neg=zeros(num*negNum,dimNum);
    end
    fea_pos((i-1)*posNum+1:i*posNum,:)=f_pos';
    fea_neg((i-1)*negNum+1:i*negNum,:)=f_neg';
    disp(i/num);
end
save(sprintf('%spart%02d_svmFeature.mat',root_out,partID),'fea_pos','fea_neg','posNum','negNum');

clear fea_pos fea_neg posNum negNum;load(sprintf('%spart%02d_svmFeature.mat',root_out,partID),'fea_pos','fea_neg','posNum','negNum');
for i=1:length(stepList)
    step=stepList(i);
    antNum=antList(i);
    pN=antNum*posNum;
    nN=antNum*negNum;
    X=[fea_pos(1:pN,:);fea_neg(1:nN,:)];
    Y=[ones(pN,1);-ones(nN,1)];
    %linearSVM=fitcsvm(X,Y,'KernelFunction','linear','BoxConstraint',10000,'Prior','uniform');
    linearSVM=fitcsvm(X,Y,'KernelFunction','RBF','Prior','uniform','KernelScale','auto');
    %linearSVM=fitcsvm(X(1:pN,:),Y(1:pN),'KernelFunction','rbf','KernelScale','auto','OutlierFraction',0.05);
    save(sprintf('%spart%02d_step%02d_svm.mat',root_out,partID,step),'linearSVM');
end
end


function f_neg=getNegF(I_patch,tar_h,tar_w,scale_h,scale_w,net)
driftList_main=[-0.6,-0.4,0.4,0.6];
driftList_other=[-0.25,0,0.25];
scaleRateList=0.9:0.35:1.6;
c=0;
for scaleRate=scaleRateList
    SH=scale_h*scaleRate;
    SW=scale_w*scaleRate;
    for d1=driftList_main
        for d2=driftList_other
            c=c+1;
            theH=tar_h+d1*scale_h;
            theW=tar_w+d2*scale_w;
            tmp=extractFC7(I_patch,net,theH,theW,SH,SW,false);
            if(c==1)
                f_neg=zeros(length(tmp),length(driftList_main)*length(driftList_other)*length(scaleRateList)*4);
            end
            f_neg(:,c*2-1)=tmp;
            tmp=extractFC7(I_patch,net,theH,theW,SH,SW,true);
            f_neg(:,c*2)=tmp;
            c=c+1;
            theH=tar_h+d2*scale_h;
            theW=tar_w+d1*scale_w;
            tmp=extractFC7(I_patch,net,theH,theW,SH,SW,false);
            f_neg(:,c*2-1)=tmp;
            tmp=extractFC7(I_patch,net,theH,theW,SH,SW,true);
            f_neg(:,c*2)=tmp;
        end
    end
end
end


function f_pos=getPosF(I_patch,tar_h,tar_w,scale_h,scale_w,net)
scaleRateList=0.9:0.1:1.3;
num=length(scaleRateList);
for c=1:num
    scaleRate=scaleRateList(c);
    tmp=extractFC7(I_patch,net,tar_h,tar_w,scale_h*scaleRate,scale_w*scaleRate,false);
    if(c==1)
        f_pos=zeros(length(tmp),num*2);
    end
    f_pos(:,c*2-1)=tmp;
    tmp=extractFC7(I_patch,net,tar_h,tar_w,scale_h*scaleRate,scale_w*scaleRate,true);
    f_pos(:,c*2)=tmp;
end
end
