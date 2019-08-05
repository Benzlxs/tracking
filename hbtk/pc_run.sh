
config='./config/kitti_tracking.config'
code='./pointcloud_tracking.py'
#output_dir='./results/tracking_2'
output_dir='./results'
mode=pointcloud_tracking
display=True
#display=False
display_trajectory=True

exe=~/miniconda3/bin/python3
#exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode --config_path=$config --output_dir=$output_dir --display=$display --display_trajectory=$display_trajectory"
$run_script
