function label_set_corner=getCorner(label_set,HWdift,cornerScale)
label_set_corner=label_set;
for i=1:length(label_set)
    label_set_corner(i).pHW_center=label_set(i).pHW_center+HWdift.*label_set(i).pHW_scale;
    label_set_corner(i).pHW_scale=label_set(i).pHW_scale.*cornerScale;
end
end
