code=eval.py
#mode=run_det_results
#mode=run_trk_results
mode=run_det_trk_results

#exe=~/miniconda3/bin/python3
exe=~/miniconda3/bin/pudb3

run_script="$exe $code $mode"
#run_script="$exe $code $mode --config_path=$config --output_dir=$output_dir"
$run_script
