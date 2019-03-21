import numpy as np
from hbtk.filters.data_association import associate_detections_to_trackers
from hbtk.filters.kalman_filter import KalmanBoxTracker

class Sort(object):
    """
    SORT: A Simple, Online and Realtime Tracker
    """
    def __init__(self, config=None, data_association="Hungarian"):
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
        
        # select the data_association algorithm
        data_association_apparoch_dic = {
                "Hungarian":associate_detections_to_trackers
                }
        self.data_association = data_association_apparoch_dic[data_association]


    def update(self, object_dets, reset_confid=True):
        """
        Args:
            object_dets:  a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]

        Returns:
            a similar array, where the last column is the object ID.

        Raises:
        
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict()[0]
          trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        # matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        matched, unmatched_dets, unmatched_trks = self.data_association(object_dets,trks)


        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(object_dets[d,:][0], reset_confid=reset_confid)

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(object_dets[i,:]) 
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # condition change with detector_config.interval_num  self.hits/self.age>0.5, the updating frequency should be over 50%, the self.hit>2, the updating times should be over a threshold
            # if((trk.time_since_update < self.age_tolerate) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            if((trk.time_since_update < self.age_tolerate) and \
                    (((trk.hits+1)/np.float(trk.age+1)>=self.ratio_hit_age and trk.hits>=self.min_hits) or self.frame_count <= self.start_hits)):
              ret.append(np.concatenate((d,[trk.id+1],[trk.confid])).reshape(1,-1)) # +1 as MOT benchmark requires positive, last item is deteciton confidence
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        if(len(ret)>0):
          return np.concatenate(ret)
        return np.empty((0,5))
 

