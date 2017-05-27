Name_batch='cow'; % the category name
partID=1; % the id of the head part
theConf=startup; % settings
step_fineTune(Name_batch,theConf); %train a CNN (fine-tune a CNN to classify a category and backgrond)
step_getAvgResponse(Name_batch,theConf); % compute statistics of the pre-trained CNN
load(sprintf('%s%s/label_set_part%02d.mat',theConf.output.dir,Name_batch,partID),'label_set');
step_batch(Name_batch,partID,label_set,theConf); % learning
[detection,normDist,~,~]=step_getResult(Name_batch,partID,theConf); % compute Accuracy
fprintf('Detection accuracy (IoU>=0.5): %.3f      normalized distance %.3f\n',detection,normDist)
