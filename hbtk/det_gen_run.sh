
tracking_config='./config/kitti_tracking.config'
detection_config='./config/detection.config'
code='./dataset/detections_gt_generation.py'
#output_dir='./results/tracking_2'
output_dir='./results'
#mode=save_detection_gt
mode=save_detection_classification
#mode=create_negative_samples

exe=~/miniconda3/bin/python3
#exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode --tracking_config_path=$tracking_config --detection_config_path=$detection_config"
$run_script
