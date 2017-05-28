function step_getAvgResponse(Name_batch,theConf)
weight.local=0.1;
weight.geo=1;

load([theConf.output.dir,Name_batch,'/net_finetune.mat'],'net','info');
if(theConf.QA.IsUseGPU)
    net=vl_simplenn_move(net,'gpu');
end
objset_neg=getNegObjSet(theConf,Name_batch);
[avg_res,avgrelu_res,sqrtvar_layer]=getAvgResponse(objset_neg,theConf,net);
objNum=length(objset_neg);
stat_all=getStat(avg_res,avgrelu_res,sqrtvar_layer,objNum);
save([theConf.output.dir,Name_batch,'/all_statistics.mat'],'stat_all');
[Loss,roughCNN]=getGenLoss_roughCNN(Name_batch,net,weight,theConf);
save([theConf.output.dir,Name_batch,'/genLoss.mat'],'Loss');
save([theConf.output.dir,Name_batch,'/roughCNN.mat'],'roughCNN');
end


function [avg_res,avgrelu_res,sqrtvar_layer]=getAvgResponse(objset,theConf,theNet)
theConf_neg=theConf;
theConf_neg.data.imgdir=theConf_neg.data.imgdir_neg;
objNum=length(objset);
layerNum=length(theConf_neg.convnet.targetLayers);
avg_res(layerNum).x=[];
avgrelu_res(layerNum).x=[];
sqrtvar_layer(layerNum).x=[];
for objID=1:objNum
    [res,~,~]=getObjFeature(objset(objID),theConf_neg,theNet,false);
    for layer=1:layerNum
        x=f(res,layer,theConf_neg);
        if(objID==1)
            avg_res(layer).x=x;
            avgrelu_res(layer).x=max(x,0);
            sqrtvar_layer(layer).x=x.^2;
        else
            avg_res(layer).x=avg_res(layer).x+x;
            avgrelu_res(layer).x=avgrelu_res(layer).x+max(x,0);
            sqrtvar_layer(layer).x=sqrtvar_layer(layer).x+x.^2;
        end
    end
    fprintf('%d/%d\n',objID,objNum)
end
for layer=1:layerNum
    avg_res(layer).x=avg_res(layer).x./objNum;
    avgrelu_res(layer).x=avgrelu_res(layer).x./objNum;
    sqrtvar_layer(layer).x=sqrt(sqrtvar_layer(layer).x./objNum-avg_res(layer).x.^2);
end
end


function stat=getStat(avg_res,avgrelu_res,sqrtvar_layer,objNum)
stat.avg_res=avg_res;
stat.avgrelu_res=avgrelu_res;
stat.sqrtvar_layer=sqrtvar_layer;
stat.objNum=objNum;
end


function [Loss,roughCNN]=getGenLoss_roughCNN(Name_batch,net,weight,theConf)
parSize=theConf.QA.parpoolSize;
load([theConf.output.dir,Name_batch,'/all_statistics.mat'],'stat_all');
theStat_all=stat_all;clear stat_all;
layerNum=length(theConf.convnet.targetLayers);
objset=readAnnotation(Name_batch,theConf);
Loss(layerNum).Gen=[];
objNum=length(objset);
roughCNN.res=cell(objNum,1);
roughCNN.res_flip=cell(objNum,1);
parSet=repmat(struct('layerNum',layerNum,'conf',theConf),[1,parSize]);
for id_start=1:parSize:objNum
    v=repmat(struct('v',cell(layerNum,1),'weight',weight),[1,parSize]);
    for par=1:min(objNum-id_start+1,parSize)
        objID=id_start+par-1;
        obj=objset(objID);
        [res,~,~]=getObjFeature(obj,theConf,net,false);
        [res_flip,~,~]=getObjFeature(obj,theConf,net,true);
        for layer=1:layerNum
            v(par).v{layer}=f_norm(res,layer,theConf,theStat_all);
        end
        roughCNN.res{objID}=roughCNN_compress(res,theConf);
        roughCNN.res_flip{objID}=roughCNN_compress(res_flip,theConf);
        clear res res_flip
    end
    parBest=repmat(struct('best',cell(layerNum,1)),[1,parSize]);
    parfor par=1:min(objNum-id_start+1,parSize)
        theWeight=v(par).weight;
        for layer=1:parSet(par).layerNum
            theV=v(par).v{layer};
            parBest(par).best{layer}=getGenL(theV.x,theV.valid,layer,parSet(par).conf,theWeight);
        end
    end
    for par=1:min(objNum-id_start+1,parSize)
        for layer=1:layerNum
            best=parBest(par).best{layer};
            if(isempty(Loss(layer).Gen))
                Loss(layer).Gen.SUM=best.SUM;
                Loss(layer).Gen.local=best.local;
                Loss(layer).Gen.geo=best.geo;
                Loss(layer).Gen.N=1;
                Loss(layer).Gen.weight=best.weight;
            else
                Loss(layer).Gen.SUM=Loss(layer).Gen.SUM+best.SUM;
                Loss(layer).Gen.local=Loss(layer).Gen.local+best.local;
                Loss(layer).Gen.geo=Loss(layer).Gen.geo+best.geo;
                Loss(layer).Gen.N=Loss(layer).Gen.N+1;
            end
        end
    end
    disp(id_start/length(objset));
end
end
