function TT= setup_tracking_animations()
x_s = 0;
TT.xt_t1= patch(0,x_s,'y'); % tracked vehicle true
TT.xv_t1= patch(0,x_s,'r'); % tracked vehicle estimate
TT.pth_t1= plot(0,x_s,'k.','markersize',2); % tracked vehicle path estimate
TT.true_t1 = plot(0,x_s,'b');
TT.obs_t1= plot(0,x_s,'r'); % tracked observations
TT.xf_t1= plot(0,0,'r+'); % estimated features
TT.cov_t1= plot(0,0,'b'); % covariance ellipses
end
