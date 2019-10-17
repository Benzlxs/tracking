# tracking

## Testing procedures
When downloading the new sequence of vedio.
1. Run ground truth using `extract_gt_tracklets.py`;
2. Creat the cropped point with function: create_reduced_point_cloud in `detections_gt_generation.py`;
3. Estimate the road plane parameters using matlab code under folder /dataset/extract_ground_planes ;
4. Generate the detection results with code `detections_gt_generation.py`;

After generating all the detection results, the tracking visualizaton function should be launched to check all the detectio results are correct.

The evaluation procedures
1. run eval.py with run_det_trk_results function;


The simulation procedures
1. to generate the function with save_detection_tracking_multi_phases in `sequence_pc_detection.py`；
2. to evaluate the function with run_det_trk_result_in_simulation in `eval.py`;
