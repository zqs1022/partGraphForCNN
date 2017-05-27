function conf=startup
conf=configurations;
addpath(genpath('./basic_functions'));
addpath(genpath('./try_functions'));
addpath(genpath('./tool'));
addpath(conf.data.devkitdir);
addpath(conf.convnet.codedir);
addpath(conf.data.readCode);
vl_setupnn;
mex ./tool/getGenDisL/getGenDisL_core_mex.cpp
mex ./tool/parent_info/getParentAdvice_mex.cpp
end
