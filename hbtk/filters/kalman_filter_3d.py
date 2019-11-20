import numpy as np
from numpy import dot, linalg
# from filterpy.kalman import KalmanFilter
import sys
very_very_small_number = sys.float_info.min


def checking_updating_size(trk_size, det_size, len_thr = 0.5, wid_thr = 0.5):
    assert len(trk_size) == len(det_size), "len of tracker and detector should be the same"
    if np.abs(trk_size[0] - det_size[0])>len_thr or \
            np.abs(trk_size[1] -det_size[1]) >wid_thr:
        return True
    else:
        return False
def fuse_probability(trk_c, det_c):
    """
    using naive bayes classifier to do fuse
    """
    assert len(trk_c) == len(det_c), "the variables shold have the same shape"
    fuse_confid = []
    normalization = sum([trk_c[i]*det_c[i] for i in range(len(trk_c))])

    if normalization == 0.0:
        normalization = sum([(trk_c[i]+very_very_small_number)*(det_c[i]+very_very_small_number) for i in range(len(trk_c))])
        print('######## One Nan data...........')
        print("tracking confidence:{}".format(trk_c))
        print("detection confidence:{}".format(det_c))

        for i in range(len(trk_c)):
            fuse_confid.append((trk_c[i]+very_very_small_number)*(det_c[i]+very_very_small_number)/normalization)
    else:
        for i in range(len(trk_c)):
            fuse_confid.append(trk_c[i]*det_c[i]/normalization)



    return fuse_confid

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def kinematic_model(X, delta_T):
    """
    [x,y,theta, v]
    """
    theta = X[2]
    v = X[3]
    X[0] += v*delta_T*np.cos(theta)
    X[1] += v*delta_T*np.sin(theta)
    return X


class ExtendKalmanBoxTracker_3D(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, det, frame_id=0, det_in_local=None):
    """
    Initialises a tracker using initial bounding box.
    Det: [class, x, y, z, l, w, h, theta]
    State: X = [x,y,theta, velocity]
    """
    #define constant velocity model
    self.X = np.array([det[1], det[2], det[7], 0])
    npm = 1 # uncertainity of initial position
    self.P =  np.array([[npm,0,0,0], [0,npm,0,0], [0,0,100*npm,0], [0,0,0,100*npm] ])
    npm = 0.05 # 0.05 # noise for process model
    self.Q = np.array([[npm,0,0,0], [0,npm,0,0], [0,0,npm,0], [0,0,0, npm] ])

    nom = 0.05 # 0.05 # uncertainy for observation model
    self.R = np.array([[ nom, 0, 0], [ 0, nom, 0], [ 0, 0, 10000*nom]])
    # do not trust heading, so the factor the heading is very small

    # observation matrix, H matrix
    self.H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    # lidar frequency is 100ms per frame
    self.delta_T = 0.1

    self.time_since_update = 0
    self.id = ExtendKalmanBoxTracker_3D.count
    ExtendKalmanBoxTracker_3D.count += 1
    self.color = [np.random.rand(), np.random.rand(), np.random.rand()]
    self.history = []
    self.tracklet_det = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.category = det[0]
    self.z_s      = det[3]
    self.length   = det[4]
    self.width    = det[5]
    self.height   = det[6]
    self.heading  = det[7]
    if len(det) >= 13:
        self.num_points = det[12]
        self.tracklet_det.append(det_in_local)
    else:
        self.num_points = None

    self.state_det = None
    # no ground truth
    if len(det) == 8:
        self.confid  = [1]  # which is ground truth
    if len(det) == 9:
        self.confid  =  [det[8]]  # confidence with one category
    if len(det) >= 12:   # detection with classification confidence
        # self.confid =  [ det[8], det[9]+det[11], det[10], det[12] ]
        self.confid = [ det[8], det[9], det[10], det[11] ]  # 4 classes
        #              [ Background, Car, Pedestrain, Cyclist ]
    # the number of counts for certain classification
    num_classes = 4  # make sure that it has the same size with confidence
    self.num_counts = np.zeros((num_classes), dtype=np.int)
    self.history.append([self.X[0], self.X[1], self.X[2]])

    # the code below is for saving all the detection confidence
    self.frame_ids_hist = [frame_id]
    self.detection_confidence_hist = [self.confid]
    self.tracker_confidence_hist = [self.confid]
    self.detection_class_hist = [self.category]
    self.tracker_class_hist = [self.category]

    # self.save_tracker_id = 0


  def update(self, det, reset_confid=True, det_in_local=None):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    # self.kf.update(convert_bbox_to_z(bbox))
    _z = np.array([det[1], det[2], det[7]])
    _y = _z - dot(self.H, self.X)
    _S = dot(dot(self.H, self.P), self.H.T) + self.R

    # map system uncertainty into kalman gain
    try:
        K = dot(dot(self.P, self.H.T), linalg.inv(_S))
    except:
        # can't invert a 1D array, annoyingly
        K = dot(dot(self.P, self.H.T), 1./_S)
    # predict new x with residual scaled by the kalman gain
    self.X = self.X + dot(K, _y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, self.H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])

    self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, self.R), K.T)


    if reset_confid:
        self.category = det[0]
        self.z_s      = det[3]
        self.length   = det[4]
        self.width    = det[5]
        self.height   = det[6]
        # no ground truth
        # if len(det) > 8:
        #    self.confid  =  det[8]
        # else:
        #    self.confid  = 1
        if len(det) == 8:
            self.confid  = [1]  # which is ground truth
        if len(det) == 9:
            self.confid  =  [det[8]]  # confidence with one category
        if len(det) == 12:   # detection with classification confidence
            #self.confid =  [ det[8], det[9]+det[11], det[10], det[12] ]
            self.confid = [det[8], det[9], det[10], det[11]]
            #              [ Background, Car, Pedestrain, Cyclist ]

        # self.num_points = det[12]
    if self.num_points is not None:
        # self.tracklet_det.append(det)
        self.tracklet_det.append(det_in_local)

    self.history.append([self.X[0], self.X[1], self.X[2]])

  def update_fusion(self, det, label_to_num, frame_id, fusion_confidence=0.98, points_threshold = 0.4, counts_threshold=28):
    """
    Updates the state vector with observed bbox.
    Adding the function of confidence fusion
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    # self.kf.update(convert_bbox_to_z(bbox))
    _z = np.array([det[1], det[2], det[7]])
    _y = _z - dot(self.H, self.X)
    _S = dot(dot(self.H, self.P), self.H.T) + self.R

    # map system uncertainty into kalman gain
    try:
        K = dot(dot(self.P, self.H.T), linalg.inv(_S))
    except:
        # can't invert a 1D array, annoyingly
        K = dot(dot(self.P, self.H.T), 1./_S)
    # predict new x with residual scaled by the kalman gain
    self.X = self.X + dot(K, _y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, self.H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])

    self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, self.R), K.T)

    # save all
    self.frame_ids_hist.append(frame_id)
    self.detection_confidence_hist.append([det[8], det[9], det[10], det[11]])
    self.detection_class_hist.append(det[0])

    if self.state_det == label_to_num.good_enough:
        self.z_s      = (det[3] + self.z_s)/2
        self.length   = (det[4] + self.length)/2
        self.width    = (det[5] + self.width)/2
        self.height   = (det[6] + self.height)/2
        self.heading  = (det[7] + self.heading)/2
        self.num_points = det[12]
        self.category = det[0]
        # test this function later
        det[0] = self.category
        det[8] = self.confid[0]
        det[9] = self.confid[1]
        det[10]= self.confid[2]
        det[11]= self.confid[3]


    else:
        # checking whether it is a new observation or not
        # if yes, fuse probability, if no, keep waiting
        #trk_size = [self.length, self.width]
        #det_size = [det[4], det[5]]
        # 0:bg, 1:car, 2, pedestrian, 3:cyclist
        #if self.category >=2:
        #    _l = 0.25
        #    _w = 0.25
        #else:
        #    # bg and car are the large objects
        #    _l = 0.8
        #    _w = 0.8
        #update_flage = checking_updating_size(trk_size, det_size, len_thr = _l, wid_thr=_w)

        trk_num_points = self.num_points
        det_num_points = det[12]
        trk_category = self.category
        # object enter the range of interest
        if trk_category>=label_to_num.unknow_object_label:
            trk_category -= label_to_num.unknow_object_label
        det_category = det[0]
        if trk_category == det_category and abs(trk_num_points-det_num_points)> 15 and self.hits >= 1:
        # if trk_category == det_category and abs(trk_num_points-det_num_points)/float(det_num_points) >points_threshold:
            update_flage = True
        else:
            update_flage = False

        # fuse the detection confidence together
        if update_flage:
            # _det_confid = [ det[8], det[9]+det[11], det[10], det[11] ]
            _det_confid = [ det[8], det[9], det[10], det[11] ]
            _fuse_confid = fuse_probability(self.confid, _det_confid)
            self.confid = _fuse_confid
            # updating other parameters
            self.category = det[0]
            self.z_s      = (det[3] + self.z_s)/2
            self.length   = (det[4] + self.length)/2
            self.width    = (det[5] + self.width)/2
            self.height   = (det[6] + self.height)/2
            self.heading  = (det[7] + self.heading)/2
            self.num_points = det[12]
            # checking whether it is good enough or not

            # pass back to det
            det[8] = self.confid[0]
            det[9] = self.confid[1]
            det[10]= self.confid[2]
            det[11]= self.confid[3]

            _max_confid = max(self.confid)
            if _max_confid > fusion_confidence:
                _ind_ = self.confid.index(_max_confid)
                self.num_counts[_ind_] +=1
                #self.state_det = label_to_num.good_enough
                #_ind_ = self.confid.index(_max_confid)
                #self.confid = [0.]*len(self.confid)
                #self.confid[_ind_] = 1.   # 100% sure
            _max_counts = max(self.num_counts)
            if _max_counts > counts_threshold:
                self.state_det = label_to_num.good_enough
                print("*********One certain classification!")
        else:
            # use the PDF of tracker to update the PDF of detector
            _det_confid = [ det[8], det[9], det[10], det[11] ]
            _fuse_confid = fuse_probability(self.confid, _det_confid)
            det[8] = _fuse_confid[0]
            det[9] = _fuse_confid[1]
            det[10]= _fuse_confid[2]
            det[11]= _fuse_confid[3]
            # self.confid = _fuse_confid   #####



    # save the class
    self.tracker_confidence_hist.append(self.confid)
    self.tracker_class_hist.append(self.category)

    check_nan = np.argwhere(np.isnan(det))
    if check_nan.size!=0:
        import pudb; pudb.set_trace()

    self.history.append([self.X[0], self.X[1], self.X[2]])

    return det

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    theta    = self.X[2]
    velocity = self.X[3]
    delta_t  = self.delta_T
    self.X   = kinematic_model(self.X, delta_t)
    # Jocobian of processing model
    J = np.array([[ 1, 0, -delta_t*velocity*np.sin(theta), delta_t*np.cos(theta)],
                  [ 0, 1,  delta_t*velocity*np.cos(theta), delta_t*np.sin(theta)],
                  [ 0, 0,                          1,            0              ],
                  [ 0, 0,                          0,            1             ]])

    self.P = dot(dot(J,self.P),J.T) + self.Q
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1

    self.history.append([self.X[0], self.X[1], self.X[2]])

    return self.X #self.history[-1]

  def get_state(self):
    """
    """
    #return convert_x_to_bbox(self.kf.x)
    return self.X


