import numpy as np
from numpy import dot, linalg
# from filterpy.kalman import KalmanFilter



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
  def __init__(self,det):
    """
    Initialises a tracker using initial bounding box.
    Det: [class, x, y, z, l, w, h, theta]
    State: X = [x,y,theta, velocity]
    """
    #define constant velocity model
    self.X = np.array([det[1], det[2], det[7], 0])
    npm = 1 # uncertainity of initial position
    self.P =  np.array([[npm,0,0,0], [0,npm,0,0], [0,0,npm,0], [0,0,0,100*npm] ])
    npm = 0.05 # noise for process model
    self.Q = np.array([[npm,0,0,0], [0,npm,0,0], [0,0,npm,0], [0,0,0,npm] ])

    nom = 0.05 # uncertainy for observation model
    self.R = np.array([[ nom, 0, 0], [ 0, nom, 0], [ 0, 0, nom]])

    # observation matrix, H matrix
    self.H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    # lidar frequency is 100ms per frame
    self.delta_T = 0.1

    self.time_since_update = 0
    self.id = ExtendKalmanBoxTracker_3D.count
    ExtendKalmanBoxTracker_3D.count += 1
    self.color = [np.random.rand(), np.random.rand(), np.random.rand()]
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.category = det[0]
    self.z_s      = det[3]
    self.length   = det[4]
    self.width    = det[5]
    self.height   = det[6]
    # no ground truth
    if len(det) > 8:
        self.confid  =  det[8]
    else:
        self.confid  = 1  # which is ground truth

    self.history.append([self.X[0], self.X[1], self.X[2]])


  def update(self, det, reset_confid=True):
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
        if len(det) > 8:
            self.confid  =  det[8]
        else:
            self.confid  = 1

    self.history.append([self.X[0], self.X[1], self.X[2]])


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
    return self.X #self.history[-1]

  def get_state(self):
    """
    """
    #return convert_x_to_bbox(self.kf.x)
    return self.X


