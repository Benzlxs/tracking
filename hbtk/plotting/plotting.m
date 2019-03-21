% plotting the results.
clf, clear
% computation 
Main_FLOPs = 670.59;  %FRCNN method
Auxi_FLOPs = 23.04;   % mobilenet SSD method
x=[1:8];
y=(Main_FLOPs+(x-1)*Auxi_FLOPs)./x;
figure(1), 
plot(x-1,y,'-r','LineWidth',2);
xlabel('Interval frame');
ylabel('FLOPs(G)')
axis([-1,9, 0, 700]);
legend('computation complexity');
% detection performance
clf,
x=[0:7];
y1=[0.816, 0.7169, 0.6950, 0.6716, 0.5930, 0.5781, 0.5614, 0.5548];
y2=[0.816, 0.5368, 0.4417,  0.4272, 0.3495,  0.3441, 0.3396, 0.3359];
figure(1), 
plot(x,y1,'-r', x,y2, '-b','LineWidth',2);
xlabel('Interval frame');
ylabel('mAP')
axis([-1,9, 0, 1]);
legend('with tracking','without tracking');

%tracking performance
clf,
x=[0:7];
y1=[40.3, 33.7, 28.4, 23.9, 20.1, 17.6, 14.8, 13.1];
%y2=[85.5, 80.6, 79.0,  78.1,77.3,  76.6, 76.1, 75.5];
figure(1), 
%plot(x,y1,'-r', x,y2, '-b','LineWidth',2);
plot(x,y1,'-r','LineWidth',2)
xlabel('Interval frame');
ylabel('tracking accuracy')
axis([-1,9, 0, 50]);
legend('MOTA tracking');