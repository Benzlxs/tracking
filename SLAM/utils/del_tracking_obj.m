function [x_trk , P_trk, count_trk, ind_trk_obj] = del_tracking_obj(x_trk , P_trk, count_trk, num_del, ind_trk_obj)
% description: deletet the tracked objects that have not been observed for a long time
% input:
%     x_trk: state of tracked object
%     P_trk: covariance of all tracked object
%     count_trk: conuters
%     num_del: the number threshold to delete the tracked object
%     ind_trk_obj: index of object in tracking set
%
% output
%     x_trk: state of tracked object
%     P_trk: covariance of all tracked object
%     count_trk: conuters 
%     ind_trk_obj: index of object in tracking set
%
% Xuesong LI, 2018

num_obj = size(x_trk,1);
del_ind = [];
for i = 1:num_obj
    if count_trk(i) >= num_del
        del_ind = [del_ind i];
    end
end

x_trk(del_ind,:) = [];
P_trk(del_ind,:,:) = [];
count_trk(del_ind) = [];
ind_trk_obj(del_ind) = [];

end