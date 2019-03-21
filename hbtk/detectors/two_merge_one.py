import numpy as np


class HybridDetector(object):
    """This class is about hybrid detector

    Attributes:
        len__: represent blablabla;
        method: detailed methods;
    """
    def __init__(self, detector_config):
        """
        Initializes a tracker using its config file
        The detection file format is as following:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        
        self.main_method = detector_config.main_method
        self.auxiliary_method = detector_config.auxiliary_method
        self.interval_num = detector_config.interval_num
        self.data_subset = detector_config.data_subset
        self.confidence = detector_config.confidence
        self.num_frames = {}  # saved as dictionary variables
        self.fused_detections = {} # saved fused detection results.
        # self checking, make sure the detection numbers are the same for both methods
        for name in self.data_subset:
            print('Reading detections from data sequence %s'%(name))
            result_main_detections = np.loadtxt('/%s/%s/det.txt'%(self.main_method, name),delimiter=',') # load the detection results from the first one
            result_auxiliary_detections = np.loadtxt('/%s/%s/det.txt'%(self.auxiliary_method, name),delimiter=' ') # load the detection results from the second one
            num_main_detection = result_main_detections[:,0].max()
            num_auxiliary_detections = result_auxiliary_detections[:,0].max()
            assert num_main_detection==num_auxiliary_detections, "the number of main and auxiliary detection methods should be the same"
            # saving the number of detection results
            self.num_frames[name] = int(num_main_detection)
            self.fused_detections[name] = np.empty((0,7), dtype=result_main_detections.dtype)  # n*6 to save all the data
            for i in range(1,int(num_main_detection+1)):
                # the first frame of self.interval_num will be from the main detections
                if i % self.interval_num == 1:
                    temp_det = result_main_detections[result_main_detections[:,0]==i,0:7]
                    temp_det = temp_det[temp_det[:,6]>self.confidence,0:7]
                    # self.fused_detections[name] = np.append(self.fused_detections[name], \
                    #                    result_main_detections[result_main_detections[:,0]==i,0:7], axis=0)
                    self.fused_detections[name] = np.append(self.fused_detections[name], temp_det, axis=0)
                else:
                    temp_det = result_auxiliary_detections[result_auxiliary_detections[:,0]==i,0:7]
                    temp_det = temp_det[temp_det[:,6]>self.confidence,0:7]
                    # self.fused_detections[name]= np.append(self.fused_detections[name], \
                    #                            result_auxiliary_detections[result_auxiliary_detections[:,0]==i,0:7], axis=0)
                    self.fused_detections[name] = np.append(self.fused_detections[name], temp_det, axis=0)
                
 
    def get_num_frames(self, name_subset):
        """
        Getting the number of frames in a subset of sequence data 
        """

        return self.num_frames[name_subset]


    def fetch_detection(self, name_subset, frame_id):
        """
        Getting the detection result from the corresponding frame_id in given subset name
        """

        subdata_detections = self.fused_detections[name_subset][:,0]
        return self.fused_detections[name_subset][subdata_detections==frame_id,:]



