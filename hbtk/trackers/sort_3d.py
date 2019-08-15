import collections
import numpy as np
from hbtk.filters.data_association import associate_detections_to_trackers, associate_detections_to_trackers_distance
from hbtk.filters.kalman_filter_3d import ExtendKalmanBoxTracker_3D

LABEL_NUM = collections.namedtuple('LABEL_NUM',['unknow_object_label', 'need_more', 'good_enough'])
label_to_num = LABEL_NUM(unknow_object_label=256, need_more=4, good_enough=8)

class Sort_3d(object):
    """
    SORT: A Simple, Online and Realtime Tracker
    """
    def __init__(self, config=None, data_association="Hungarian_dist", fusion_confidence=0.98):
        """
        Sets key parameters for SORT
        Args:
            max_age:
            min_hits: 1 updating is 1 hit
            age_tolerate: 1 prediction is 1 age, this is how many age tolerate.
            data_association: the name of data association method;
        Raises:
        """
        self.max_age = config.max_age
        self.min_hits = config.min_hits
        self.trackers = []
        self.frame_count = 0
        self.age_tolerate = config.age_tolerate
        self.start_hits = config.start_hits
        self.ratio_hit_age = config.ratio_hit_age
        # the range of interest
        self.interest_range = config.interest_range
        self.fusion_confidence = fusion_confidence

        # select the data_association algorithm
        data_association_apparoch_dic = {
                "Hungarian":associate_detections_to_trackers,
                "Hungarian_dist": associate_detections_to_trackers_distance,
                }
        self.data_association = data_association_apparoch_dic[data_association]


    def update(self, object_dets, reset_confid=True):
        """
        Args:
            object_dets:  a numpy array of detections in the format
            [class, x, y, z, l, w, h, theta]

        Returns:
            a similar array, where the last column is the object ID.

        Raises:

        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict() #[x,y,theta]
          trk[:] = [pos[0], pos[1], pos[2], pos[3]]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        # matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        matched, unmatched_dets, unmatched_trks = self.data_association(object_dets, trks)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(object_dets[d,:][0], reset_confid=reset_confid)

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = ExtendKalmanBoxTracker_3D( object_dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # condition change with detector_config.interval_num  self.hits/self.age>0.5, the updating frequency should be over 50%, the self.hit>2, the updating times should be over a threshold
            # if((trk.time_since_update < self.age_tolerate) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            if((trk.time_since_update < self.age_tolerate) and \
                    (((trk.hits+1)/np.float(trk.age+1)>=self.ratio_hit_age and trk.hits>=self.min_hits) or self.frame_count <= self.start_hits)):
              ret.append(np.concatenate((d,[trk.id+1],[max(trk.confid)])).reshape(1,-1)) # +1 as MOT benchmark requires positive, last item is deteciton confidence
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        if(len(ret)>0):
          return np.concatenate(ret)
        return np.empty((0,5))


    def update_range(self, object_dets, robot_loc, reset_confid=True):
        """
        Args:
            object_dets:  a numpy array of detections in the format
            [class, x, y, z, l, w, h, theta]

        Returns:
            a similar array, where the last column is the object ID.

        Raises:

        """
        num_classification_run = 0
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict() #[x,y,theta]
          trk[:] = [pos[0], pos[1], pos[2], pos[3]]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        # matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        matched, unmatched_dets, unmatched_trks = self.data_association(object_dets, trks)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            _distance = np.sqrt((trk.X[0] - robot_loc[0])**2 + \
                                (trk.X[1] - robot_loc[1])**2)
            # trk enters the interested range and without classification, thus,
            # run classification model
            if _distance < self.interest_range and int(trk.category)== label_to_num.unknow_object_label:
                num_classification_run +=1

            trk.update(object_dets[d,:][0], reset_confid=reset_confid)

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # check whether distance is within range or not
            _distance = np.sqrt((object_dets[i,1] - robot_loc[0])**2 + \
                                (object_dets[i,2] - robot_loc[1])**2)
            # the unknow object label
            if _distance > self.interest_range:
                object_dets[i,0] = label_to_num.unknow_object_label
            else:
                num_classification_run += 1

            trk = ExtendKalmanBoxTracker_3D( object_dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # condition change with detector_config.interval_num  self.hits/self.age>0.5, the updating frequency should be over 50%, the self.hit>2, the updating times should be over a threshold
            # if((trk.time_since_update < self.age_tolerate) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            if((trk.time_since_update < self.age_tolerate) and \
                    (((trk.hits+1)/np.float(trk.age+1)>=self.ratio_hit_age and trk.hits>=self.min_hits) or self.frame_count <= self.start_hits)):
              ret.append(np.concatenate((d,[trk.id+1], [max(trk.confid)])).reshape(1,-1)) # +1 as MOT benchmark requires positive, last item is deteciton confidence
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        if(len(ret)>0):
          return np.concatenate(ret), num_classification_run
        return np.empty((0,5)), num_classification_run


    def update_range_fusion(self, object_dets, robot_loc, reset_confid=True):
        """
        Args:
            object_dets:  a numpy array of detections in the format
            [class, x, y, z, l, w, h, theta]

        Returns:
            a similar array, where the last column is the object ID.

        Raises:

        """
        num_classification_run = 0
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict() #[x,y,theta]
          trk[:] = [pos[0], pos[1], pos[2], pos[3]]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        # matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        matched, unmatched_dets, unmatched_trks = self.data_association(object_dets, trks)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            _distance = np.sqrt((trk.X[0] - robot_loc[0])**2 + \
                                (trk.X[1] - robot_loc[1])**2)
            # trk enters the interested range and without classification, thus,
            # run classification model
            if _distance < self.interest_range and int(trk.category) == label_to_num.unknow_object_label:
                num_classification_run +=1

            trk.update_fusion(object_dets[d,:][0], label_to_num,  fusion_confidence = self.fusion_confidence)


        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # check whether distance is within range or not
            _distance = np.sqrt((object_dets[i,1] - robot_loc[0])**2 + \
                                (object_dets[i,2] - robot_loc[1])**2)
            # the unknow object label
            if _distance > self.interest_range:
                object_dets[i,0] = label_to_num.unknow_object_label
            else:
                num_classification_run += 1

            # give different label
            trk = ExtendKalmanBoxTracker_3D( object_dets[i,:])
            _max_confid = max(trk.confid)
            if _max_confid < self.fusion_confidence:
                # need more data
                trk.state_det = label_to_num.need_more
            else:
                # good enough, no classification anymore
                trk.state_det = label_to_num.good_enough
                _ind_ = trk.confid.index(_max_confid)
                trk.confid = [0.]*len(trk.confid)
                trk.confid[_ind_] = 1.   # 100% sure

            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # condition change with detector_config.interval_num  self.hits/self.age>0.5, the updating frequency should be over 50%, the self.hit>2, the updating times should be over a threshold
            # if((trk.time_since_update < self.age_tolerate) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            if((trk.time_since_update < self.age_tolerate) and \
                    (((trk.hits+1)/np.float(trk.age+1)>=self.ratio_hit_age and trk.hits>=self.min_hits) or self.frame_count <= self.start_hits)):
              ret.append(np.concatenate((d,[trk.id+1], [max(trk.confid)])).reshape(1,-1)) # +1 as MOT benchmark requires positive, last item is deteciton confidence
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        if(len(ret)>0):
          return np.concatenate(ret), num_classification_run
        return np.empty((0,5)), num_classification_run
