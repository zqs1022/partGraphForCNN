function step_fineTune(Name_batch,theConf)
mkdir([theConf.output.dir,Name_batch,'/']);
objset=readAnnotation(Name_batch,theConf);
objset_neg=getNegObjSet(theConf,Name_batch);
[net,info]=finetune(theConf,objset,objset_neg,Name_batch);
net.layers{end}.type='softmax';
save([theConf.output.dir,Name_batch,'/net_finetune.mat'],'net','info');
end
