import numpy as np
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
small_number = np.finfo(np.float32).tiny

@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



def associate_detections_to_trackers_distance(detections,trackers,dist_threshold = 4.5): #0.9 max_speed = 1.2*10*3.6 ~ 40Km/h
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Args:
      detections: [class, x, y, z, l, w, h, theta, config]
      tracks: [x, y, theta]
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  _dets = detections[:,[1,2,7,4,5,6,12]]
  _trks = np.array(trackers)

  _dets = np.expand_dims(_dets, axis=1)
  _trks = np.expand_dims(_trks, axis=0)

  # iou_matrix = (_dets[:, :, 0] - _trks[:, :, 0])**2 + (_dets[:, :, 1] - _trks[:, :, 1])**2  + (_dets[:, :, 2] - _trks[:, :, 2])**2
  # cutting off the heading error
  # distance difference
  dist_matrix = (_dets[:, :, 0] - _trks[:, :, 0])**2 + (_dets[:, :, 1] - _trks[:, :, 1])**2 #  + (_dets[:, :, 2] - _trks[:, :, 2])**2
  # iou_matrix += small_number
  # size difference
  size_matrix = (_dets[:, :, 3] - _trks[:, :, 4])**2 + (_dets[:, :, 4] - _trks[:, :, 5])**2 + (_dets[:, :, 5] - _trks[:, :, 6])**2
  # velocity difference
  numb_matrix = abs(_dets[:, :, 6] - _trks[:, :, 7])

  dist_weight = 2.5   # 2   # 2.5
  size_weight = 2.5   # 3   # 2.5
  numb_weight = 0.1  # 0.1 # 0.1

  iou_matrix = dist_weight*np.sqrt(dist_matrix) + size_weight*np.sqrt(size_matrix) + numb_weight*numb_matrix


  # iou_matrix = np.divide(1., iou_matrix)
  #for d,det in enumerate(detections):
  #  for t,trk in enumerate(trackers):
  #    iou_matrix[d,t] = np.sqrt(np.square)
  # matched_indices = linear_assignment(-iou_matrix)

  matched_indices = linear_assignment(iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)

  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with too large distance
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]] >  dist_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



def associate_detections_to_trackers_distance_bk(detections,trackers,dist_threshold = 0.4): #0.9
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Args:
      detections: [class, x, y, z, l, w, h, theta, config]
      tracks: [x, y, theta]
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  _dets = detections[:,[1,2,7]]
  _trks = np.array(trackers)

  _dets = np.expand_dims(_dets, axis=1)
  _trks = np.expand_dims(_trks, axis=0)

  iou_matrix = (_dets[:, :, 0] - _trks[:, :, 0])**2 + (_dets[:, :, 1] - _trks[:, :, 1])**2  + (_dets[:, :, 2] - _trks[:, :, 2])**2
  # cutting off the heading error
  iou_matrix = (_dets[:, :, 0] - _trks[:, :, 0])**2 + (_dets[:, :, 1] - _trks[:, :, 1])**2 #  + (_dets[:, :, 2] - _trks[:, :, 2])**2
  iou_matrix += small_number
  iou_matrix = np.sqrt(iou_matrix)

  iou_matrix = np.divide(1., iou_matrix)

  #for d,det in enumerate(detections):
  #  for t,trk in enumerate(trackers):
  #    iou_matrix[d,t] = np.sqrt(np.square)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]< dist_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)




