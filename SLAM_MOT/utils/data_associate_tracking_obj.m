function [zf,idf, zn, zn_ind]= data_associate_tracking_obj(rob_x, obj_x, P, z, z_ind, R, gate1, gate2)
% 
% Simple gated nearest-neighbour data-association. No clever feature
% caching tricks to speed up association, so computation is O(N), where
% N is the number of features in the state.
%  Input:
%         rob_x: state of robot
%         obj_x: state of tracked moving object
%         P    : covariance of tracked moving object
%         z    : observation
%   Output:
%         zf:  associated observation
%         idf: associated index of existed object
%         zn:  new features
%
% Xuesong LI, 2018.

zf= []; zn= [];
idf= []; 
zn_ind = [];

%Nf= length(obj_x)/N_s; % number of features already in map
Nf= size(obj_x,1); % number of objects already in map

% linear search for nearest-neighbour, no clever tricks (like a quick
% bounding-box threshold to remove distant features; or, better yet,
% a balanced k-d tree lookup). TODO: implement clever tricks.
z_num = size(z,2);
for i=1:z_num
    jbest= 0;
    nbest= inf;
    outer= inf;
    
    % search for neighbours
    for j=1:Nf
        [nis, nd]= compute_association_tracking_obj(rob_x, obj_x(j,:),squeeze(P(j,:,:)),z(:,i),R);
        if nis < gate1 & nd < nbest % if within gate, store nearest-neighbour
            nbest= nd;
            jbest= j;
        elseif nis < outer % else store best nis value
            outer= nis;
        end
    end
    
    % add nearest-neighbour to association list
    if jbest ~= 0
        zf=  [zf  z(:,i)];
        idf= [idf jbest];
    elseif outer > gate2 % z too far to associate, but far enough to be a new feature
        zn= [zn z(:,i)];
        zn_ind = [zn_ind  z_ind(i)];
    end
end

function [nis, nd]= compute_association_tracking_obj(rob_x, obj_x, P, z, R)
%
% return normalised innovation squared (ie, Mahalanobis distance) and normalised distance
[zp,H]= observe_model_tracking_obj(rob_x , obj_x);
%H(2,:) = [];
v= z-zp; 
%v(2) = [];
v(2)= pi_to_pi(v(2));


nd = abs(v(1));   % distance difference
nis  = (abs(v(1)) + abs(v(2)))/2;  % angle difference
%S= H*P*H' + R(1,1);
% 
% S= H*P*H' + R;
% 
% nis= v'*inv(S)*v;
% nd= nis + log(det(S));
