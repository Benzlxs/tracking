
config='./config/kitti_tracking.config'
code='./pointcloud_tracking.py'
#output_dir='./results/tracking_2'
#output_dir='./results/range_distance'
#mode=pointcloud_tracking_gt
#mode=pointcloud_tracking_det
#mode=pointcloud_tracking_classification
#mode=pointcloud_tracking_within_one_range
#mode=pointcloud_tracking_within_range
#mode=pointcloud_tracking_within_one_range_with_fusion
mode=pointcloud_tracking_within_one_range_with_fusion_multiple
display=False
display_trajectory=False

exe=~/miniconda3/bin/python3
#exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode --config_path=$config --output_dir='./results' --display=$display --display_trajectory=$display_trajectory"
#run_script="$exe $code $mode --config_path=$config --output_dir=$output_dir"
$run_script
