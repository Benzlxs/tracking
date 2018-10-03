function [x , P, count_trk, ind_trk_obj, table] = del_slam_mot_known(x , P, count_trk, num_del, ind_trk_obj, num_lm, table)
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

num_mot = (size(x,1) - 3 - num_lm*2)/4;
assert(num_mot==length(count_trk),'the number of moving object should be the same');
del_ind = [];
del_x = [];
for i = 1:num_mot
    if count_trk(i) >= num_del
        del_ind = [del_ind i];
        rng = (3+2*num_lm+4*(i-1)+1):(3+2*num_lm+4*(i-1)+4);
        del_x = [del_x, rng];
    end
end

x(del_x,:) = [];
P(del_x,:) = [];
P(:,del_x) = [];
count_trk(del_ind) = [];
ind_trk_obj(del_ind) = [];

% deleting the moving object from mot_table
acum = 0;
nn = length(del_ind);
for j = 1:nn
    del_ind(j) = del_ind(j) - acum;
    table(find(table == del_ind(j))) = 0;
    table(find(table>del_ind(j))) = table(find(table>del_ind(j))) - 1;    
    acum = acum + 1;
end


end