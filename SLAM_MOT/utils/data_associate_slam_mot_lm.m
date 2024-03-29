function [zf,idf, zn]= data_associate_slam_mot_lm(x,P,z,R, gate1, gate2, num_lm)
% 
% Simple gated nearest-neighbour data-association. No clever feature
% caching tricks to speed up association, so computation is O(N), where
% N is the number of features in the state.
%
% Tim Bailey 2004.

zf= []; zn= [];
idf= []; 

Nxv= 3; % number of vehicle pose states
Nf= num_lm;%(length(x) - Nxv)/2; % number of features already in map

P_lm = P(1:(3+2*Nf),1:(3+2*Nf)); %% covariance is too large
x_lm = x(1:(3+2*Nf));
% linear search for nearest-neighbour, no clever tricks (like a quick
% bounding-box threshold to remove distant features; or, better yet,
% a balanced k-d tree lookup). TODO: implement clever tricks.
for i=1:size(z,2)
    jbest= 0;
    nbest= inf;
    outer= inf;
    
    % search for neighbours
    for j=1:Nf
        [nis, nd]= compute_association(x_lm,P_lm,z(:,i),R, j);
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
    end
end

function [nis, nd]= compute_association(x,P,z,R,idf)
%
% return normalised innovation squared (ie, Mahalanobis distance) and normalised distance
[zp,H]= observe_model(x, idf);
v= z-zp; 
v(2)= pi_to_pi(v(2));

% nd = abs(v(1));   % distance difference
% nis  = (abs(v(1)) + abs(v(2)))/2;  % angle difference
S= H*P*H' + R;
 
nis= v'*inv(S)*v;
nd= nis + log(det(S));
