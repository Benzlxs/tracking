
config='./config/tracking_1.config'
code='./tracking.py'
output_dir='./results'
mode=tracking

#exe=~/miniconda3/bin/python3
exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode --config_path=$config --output_dir=$output_dir"
$run_script
