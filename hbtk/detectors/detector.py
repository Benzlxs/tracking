import numpy as np


class HybridDetectori(object):
    """This class is about hybrid detector

    Attributes:
        len__: represent blablabla;
        method: detailed methods;
    """
    def __init__(self, detector_config):
        """
        Initializes a tracker using its config file
        """
        
        self.main_method = detector_config.main_method
        self.auxiliary_method = detector_config.auxiliary_method
        self.interval_num = detector_config.interval_num
        self.data_subset = detector_config.data_subset
        self.num_frames = {}  # saved as dictionary variables
        self.fused_detections = {} # saved fused detection results.
        # self checking, make sure the detection numbers are the same for both methods
        for name in self.data_subset:
            print('Reading detections from data sequence%s'%(name))
            main_detections = np.loadtxt('/%s/%s/det/det.txt'%(self.main_method, name),delimiter=',') # load the detection results from the first one
            auxiliary_detections = np.loadtxt('/%s/%s/det/det.txt'%(self.auxiliary_method, name),delimiter=',') # load the detection results from the second one
            num_main_detection = main_detections[:,0].max()
            num_auxiliary_detections = auxiliary_detections[:,0].max()
            assert num_main_detection==num_auxiliary_detections, "the number of main and auxiliary detection methods should be the same"
            # saving the number of detection results
            self.num_frames[name] = num_main_detection
            self.fused_detections[name] = main_detections
            for i in range(1,(num_main_detection+1)):
                # the first frame of self.interval_num will be from the main detections
                if i % self.interval_num!=1:
                    self.fused_detections[name][auxiliary_detections[:,0]==i,:] = 
                                                auxiliary_detections[auxiliary_detections[:,0]==i,:]
                



    
    def get_num_frames(self, name_subset):
        """
        Getting the number of frames in a subset of sequence data 
        """

        return self.num_frames[name_subset]


    def fetch_detection(self, name_subset, frame_id):
        """
        Getting the detection result from the corresponding frame_id in given subset name
        """

        subdata_detections = self.fused_detections[name_subset]
        return self.fused_detections[name_subset][subdata_detections==frame_id,:]



