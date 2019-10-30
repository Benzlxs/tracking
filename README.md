# Tracking

## Testing procedures
When downloading the new sequence of vedio.
1. Run ground truth using `extract_gt_tracklets.py`;
2. Creat the cropped point with function: create_reduced_point_cloud in `detections_gt_generation.py`;
3. Estimate the road plane parameters using matlab code under folder /dataset/extract_ground_planes ;
4. Generate the detection results with code `detections_gt_generation.py`;

After generating all the detection results, the tracking visualizaton function should be launched to check all the detectio results are correct.

### The evaluation procedures
1. run eval.py with run_det_trk_results function;

### Save the detection and tracking results in simulation:
1. save without running the detector again, save_detection_tracking_multi_phases_without_detection in `sequence_pc_detection.py`;
2. save with running the detector, save_detection_tracking_multi_phases in `sequence_pc_detection.py` ;

### Accuracy in the simulation procedures
1. to generate the function with save_detection_tracking_multi_phases in `sequence_pc_detection.py`；
2. to evaluate the function with run_det_trk_result_in_simulation in `eval.py`;


### Efficiency analysis in simulation:
1. efficiency_analysis_all_phases() and efficiency_analysis_all_phases_with_fusion_model() in `eval.py`;


### To analyze the efficiency and accuracy results with real tracker
1. regenerating the detection results with the model used for similuation using the code `detections_gt_generation.py`;
2. generating the tracklet with tracker and save them under xxxx_sync/detection/tracklet_det, with function pointcloud_tracking_classification_with_saving_tracklets() in pointcloud_tracking.py;
3. running tracker to generate tracking results and put them under xxxx_sync/detection/tracklet_trk, with code function fusion_model_with_sort_results in tracker/fusion_model_sort.py;
4. producing the accuracy comparison with the function run_det_trk_result_with_real_tracker() in eval.py;
5. producing the efficiency comparison with the function
