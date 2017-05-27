Growing Interpretable Part Graphs on ConvNets via Multi-Shot Learning

Quanshi Zhang, Ruiming Cao, Ying Nian Wu, and Song-Chun Zhu in AAAI 2017

This paper proposes a learning strategy that extracts object-part concepts from a pre-trained convolutional neural network (CNN), in an attempt to 1) explore explicit semantics hidden in CNN units and 2) gradually grow a semantically interpretable graphical model on the pre-trained CNN for hierarchical object understanding. Given part annotations on very few (e.g., 3—12) objects, our method mines certain latent patterns from the pre-trained CNN and associates them with different semantic parts. We use a four-layer And-Or graph to organize the mined latent patterns, so as to clarify their internal semantic hierarchy. Our method is guided by a small number of part annotations, and it achieves superior performance (about 13%—107% improvement) in part center prediction on the PASCAL VOC and ImageNet datasets.



1. Download the matconvnet from http://www.vlfeat.org/matconvnet/ and install the matconvnet to the ./matconvnet folder.

2. Download the pre-trained VGG-16 network from http://www.vlfeat.org/matconvnet/pretrained/ and move the "imagenet-vgg-verydeep-16.mat" file to the ./matconvnet/ folder. We have tested the 1.0-beta24 version of the matconvnet. Now the ./matconvnet folder contains "doc" "examples" "README.md" "imagenet-vgg-verydeep-16.mat" and etc.

3. Download the VOC Part Dataset (the "trainval.tar.gz" file) from http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html and uncompress the file to the ./data/VOC_part/ folder. Download Pascal VOC 2010 images from http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar and uncompress the VOCdevkit folder to the ./data/VOC_part/ folder (now, the ./data/VOC_part/ folder contains "truth.bird.mat" "Annotation_part" "mat2map.m" "VOCdevkit" and etc.).

4. Uncompress the "neg.zip" file in the ./data/neg folder (now, the ./data/neg folder contains 1000 images 00001.JPEG, 00002.JPEG, and etc.).

5. run the matlab code

   cd code

   to_run.m


Please see to_run.m for detailed explanation.




Citation

@article{partGraphForCNN,
author = {Quanshi Zhang and Ruiming Cao and Ying Nian Wu and Song-Chun Zhu},

title = {Growing Interpretable Part Graphs on ConvNets via Multi-Shot Learning},

journal = {In AAAI},

volume = {},

number = {},

pages = {},

year = {2017}

}

Please see https://sites.google.com/site/cnnsemantics/ for details.
