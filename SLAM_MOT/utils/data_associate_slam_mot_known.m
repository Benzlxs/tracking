function [zf,idf, zn, zn_ind, table]= data_associate_slam_mot_known( x, z_mot_obj, ind_mot_z, num_lm, table )
% 
% For simulations with known data-associations, this function maintains
% a feature/observation lookup table. It returns the updated table, the
% set of associated observations and the set of observations to new features.
%
% Xuesong LI, 2018.

zf= []; zn= [];
idf= []; 
zn_ind = [];

idn = [];


for i = 1:length(ind_mot_z)
  ii = ind_mot_z(i);  
  if table(ii) == 0 % new features
    zn = [zn z_mot_obj(:,i)];
    idn = [idn ii];
    zn_ind = [zn_ind ii];
  else     % already existed features
    zf = [zf z_mot_obj(:,i)];
    idf = [idf table(ii)];
  end
end

Nxv = 3;
mot_num =  (size(x,1) - Nxv - num_lm*2)/4;
table(idn) = mot_num + (1:size(zn,2));



