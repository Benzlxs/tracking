
config='./config/tracking_2.config'
code='./tracking.py'
output_dir='./results/tracking_2'
mode=tracking
#display=True 
display=False
exe=~/miniconda3/bin/python3
#exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode --config_path=$config --output_dir=$output_dir --display=$display"
$run_script
