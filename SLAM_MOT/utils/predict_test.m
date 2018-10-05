function [x,P]= predict_test(x,P,v,g,Q,WB,dt)


n = size(x,1);
J =  eye(n);


s= sin(g+x(3)); c= cos(g+x(3));
vts= v*dt*s; vtc= v*dt*c;

% jacobians   
J(1:3,1:3)= [1 0 -vts;
     0 1  vtc;
     0 0 1];
Gu= [dt*c -vts;
     dt*s  vtc;
     dt*sin(g)/WB v*dt*cos(g)/WB];
  
% predict covariance
P = J*P*J';
P(1:3,1:3)= P(1:3,1:3) + Gu*Q*Gu';



% predict state
x(1:3)= [x(1) + vtc; 
         x(2) + vts;
         pi_to_pi(x(3)+ v*dt*sin(g)/WB)];

