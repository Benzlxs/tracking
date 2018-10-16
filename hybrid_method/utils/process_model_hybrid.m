% process model of moving object

function [X P] = process_model_hybrid(X, P , dt, Q_trk_ed, Q_trk_cd, PATTEN)
%{
    Function description, proces model for moving targets
    input:
        X: states of all moving targets
        P: covariance of all moving targets
    output:
        X: states of all moving targets
        P: covariance of all moving targets
%}

num_objects = size(X,1);
for i = 1: num_objects
    if PATTEN(i) == 0    % for expensive processing model  
        s= sin(X(i,3)); c= cos(X(i,3));
        vts= X(i,4)*dt*s; vtc= X(i,4)*dt*c;
        ct = c*dt; st = dt*s;
        X(i,1:4)  = [X(i,1) + vtc; 
                     X(i,2) + vts;
                     pi_to_pi(X(i,3));
                     abs(X(i,4))];
        J = [1 0 -vts ct;
             0 1  vtc st;
             0 0   1   0;
             0 0   0   1];
         P(i,:,:) = J*squeeze(P(i,:,:))*J' + dt^2*Q_trk_ed;
    else
        s= sin(X(i,3)); c= cos(X(i,3));
        vts= X(i,4)*dt*s; vtc= X(i,4)*dt*c;
        ct = c*dt; st = dt*s;
        X(i,1:4)  = [X(i,1) + vtc; 
                     X(i,2) + vts;
                     pi_to_pi(X(i,3));
                     abs(X(i,4))];
        J = [1 0 -vts ct;
             0 1  vtc st;
             0 0   1   0;
             0 0   0   1];
         P(i,:,:) = J*squeeze(P(i,:,:))*J' + dt^2*Q_trk_cd;
    end
    
end


end