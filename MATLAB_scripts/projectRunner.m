% Demo:  finding object tangent lines through given eye
addpath('/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/MatlabFiles/3d');
addpath('/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/MatlabFiles/cylalign');

% Sample image cam3
fname3 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam3/baby_cream_1_cam3.bmp';
im3 = imread(fname3);
img3 = rgb2gray(im3);

% Sample image cam2
fname2 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam2/baby_cream_1_cam2.bmp';
im2 = imread(fname2);
img2 = rgb2gray(im2);

% Sample image cam1
fname1 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam1/baby_cream_1_cam1.bmp';
im1 = imread(fname1);
img1 = rgb2gray(im1);

% Binarized image cam3
fnameBin3 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Binarized/Cam3/baby_cream_1_cam3.bmp';
mask3 = imread(fnameBin3);
dtm3 = bwdist(mask3)-bwdist(~mask3);

% Binarized image cam2
fnameBin2 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Binarized/Cam2/baby_cream_1_cam2.bmp';
mask2 = imread(fnameBin2);
dtm2 = bwdist(mask2)-bwdist(~mask2);  

% Binarized image cam1
fnameBin1 = '/Users/27836/Documents/Fourth_Year/Final_Year_Project/ResearchProject/Project_Pics/JJ_Baby_Cream/Binarized/Cam1/baby_cream_1_cam1.bmp';
mask1 = imread(fnameBin1);
dtm1 = bwdist(mask1)-bwdist(~mask1);  

% Specify object
objScale = 1.13;
rfz = @(z) 0*z + objScale*45.5;
rpfz = @(z) 0*z + 0;
% All but top contour
printRatio = 59/76;
zv = 0:objScale*1/1.5:objScale*76;  rv = rfz(zv); rpv = rpfz(zv);
thv = [0:4:360]*pi/180; % 90 evenly-spaced angles from 0 to 360
csthv = [cos(thv); sin(thv)];
Xmo = zeros(length(zv),length(thv),3);

for i=1:length(zv)
  z = zv(i);  rz = rv(i);
  if (i > printRatio*length(zv))
      rz = objScale*43.5;
  end
  Xv = [rz*cos(thv);  rz*sin(thv);  z*ones(size(thv))];
  Xmo(i,:,:) = reshape(Xv',1,[],3);
end
Xv2 = reshape(Xmo, [], 3);
Xv2 = Xv2';
% Object pose in camera 3
thx = 90*pi/180;  
Rx = [  1 0 0; 
        0 cos(thx) -sin(thx); 
        0 sin(thx) cos(thx)];
    
thy = 0*pi/180;
Ry = [  cos(thy) 0 sin(thy); 
        0 1 0; 
        -sin(thy) 0 cos(thy)];
    
thz = 0*pi/180;  
Rz = [  cos(thz) -sin(thz) 0; 
        sin(thz) cos(thz) 0; 
        0 0 1];
    
R = Rz*Ry*Rx;  
zDist = 290;
% Initial pose from bounding rect centre
% Best result: objScale = 1.13, printRatio = 59/76; zDist = 305*objScale; 
% t = [-5.4535; 34.5171*objScale; zDist-45.5*objScale];
% Or: t = [-5.4535; 39; zDist]; objScale = 1.13, printRatio = 59/76; 
% zDist = 290;
t = [-5.4535; 39; zDist];
H = [R t; 0 0 0 1];

% Eye location
imd = [1000 800];
xe = [0; 0; 0];  % eye at origin in cam2 frame

% Camera intrinsics

% Camera 3 intrinsics
K3 = cameraParams1Cam3.IntrinsicMatrix';
% Camera 2 intrinsics
K2 = cameraParams1Cam2.IntrinsicMatrix';
% Camera 1 intrinsics
K1 = cameraParams1Cam1.IntrinsicMatrix';
    
% Camera 3
% Get points on occluding contour in object frame
xeo = hom_transform(inv(H),xe);
[Xp,Xn,Xpd,Xnd] = gccyl_ocontp(zv,rv,rpv,xeo,0);
Xpn = [Xp Xn];  Xpnd = [Xpd Xnd];

% Projected points on occluding contour
P3 = K3*[eye(3) zeros(3,1)]*H;  % camera 3in object frame
xvc = hom_transform(P3,reshape(Xmo,[],3)');
xmc = reshape(xvc',[length(zv) length(thv) 2]);
xpn = hom_transform(P3,Xpn);  xpnd = hom_transform(P3,Xpnd);
xpnn = xpnd - xpn;  xpnn = xpnn./repmat(sqrt(sum(xpnn.^2)),2,1);  % unit normal

% Disk object at z=zd
zd = objScale*76;  zdr = objScale*43.5;  
dths = -65;  dthe = 65;  dnp = 30;  dnlp = 30;
thref = atan2(xeo(2),xeo(1))*180/pi;
thds = thref + dths;  thde = thref + dthe;  
while thde<thds, thde = thde + 360; end
thv = linspace(thds,thde,dnlp)/180*pi;
thva = [thv 2*thv(end)-thv(end-1)];  % augmented final point
Xdva = [zdr*cos(thva); zdr*sin(thva); zd*ones(size(thva))];
xdva = hom_transform(P3,Xdva);  xdv = xdva(:,1:end-1);

% Generate set of points on dist at approximately equal spacing in image
xpdv = diff(xdva,1,2);
xpdvl = sqrt(sum(xpdv(:,1:end-1).^2));
csl = cumsum([0 xpdvl]);
%figure;  plot(csl,thv);  axis tight;  return;
cslp = linspace(csl(1),csl(end),dnp);
thvp = interp1(csl,thv,cslp);
Xdvp = [zdr*cos(thvp); zdr*sin(thvp); zd*ones(size(thvp))];
xdvp = hom_transform(P3,Xdvp);
xpdvp = [interp1(thv,xpdv(1,:),thvp); interp1(thv,xpdv(2,:),thvp)];
xpnp = [xpdvp(2,:); -xpdvp(1,:)];  xpnp = xpnp./repmat(sqrt(sum(xpnp.^2)),2,1);

%% New code

% Camera matrices (Camera 3 is the reference camera)
Rt0Cam3 = [R t];
Rt0Cam3Homog = [Rt0Cam3; 0 0 0 1];

P3 = [K3 [0;0;0]];  % reference camera 3: R = I, t = 0
% Camera 2
RtC3toC2 = [stereoParams1Cams32.RotationOfCamera2' stereoParams1Cams32.TranslationOfCamera2'];
RtC3toC2Homog = [RtC3toC2; 0 0 0 1];
Rt0Cam2 = RtC3toC2*Rt0Cam3Homog;
Rt0Cam2Homog = [Rt0Cam2; 0 0 0 1];
P2 = K2*RtC3toC2;
% Camera 1
RtC2toC1 = [stereoParams1Cams21.RotationOfCamera2' stereoParams1Cams21.TranslationOfCamera2'];
RtC2toC1Homog = [RtC2toC1; 0 0 0 1];
Rt0Cam1 = RtC2toC1*Rt0Cam2Homog;
P1 = K1*RtC2toC1*RtC3toC2Homog;

Xv = hom_transform(H, Xv2);
%fh = figure;
%subplot(2,2,1);  imagescg(mask3);  title('Cam3');
%subplot(2,2,2);  imagescg(mask2);  title('Cam2');
%subplot(2,2,3);  imagescg(mask1);  title('Cam1');

xvC3 = hom_transform(P3,Xv);  
xvC2 = hom_transform(P2,Xv);
xvC1 = hom_transform(P1,Xv);
%subplot(2,2,1);  hold on;  shC3 = scatter(xvC3(1,:),xvC3(2,:),'r.');
%subplot(2,2,2);  hold on;  shC2 = scatter(xvC2(1,:),xvC2(2,:),'r.');
%subplot(2,2,3);  hold on;  shC1 = scatter(xvC1(1,:),xvC1(2,:),'r.');

mm = cylalignm;

rv = objScale*45.5 + 0*zv;
topIndex = round(printRatio*length(zv), 0);
% Top part of cylinder has smaller radius
rv(topIndex:end) = objScale*43.5;

mm = mm.addgcyl(zv,rv,[]);

% Top contour
zd = objScale*76;  zdr = objScale*43.5;  
mm = mm.addbdisk(zd,zdr,[dths dthe],dnp,dnlp);

mm = mm.setupview(K3,Rt0Cam3);
%mm.draw(mask3, K3, Rt0Cam3, 2);

mm = mm.setupview(K2,Rt0Cam2);
%mm.draw(mask2, K2, Rt0Cam2, 2);

mm = mm.setupview(K1,Rt0Cam1);
%mm.draw(mask1, K1, Rt0Cam1, 2);

% Perform pose optimisation by stacking errors from each frame
% Timing tests
tic, [RtCam3, residualNorm, residuals] = mm.minfit(dtm3,K3,Rt0Cam3,dtm2,K2,RtC3toC2,dtm1,K1,RtC2toC1); toc

% Accuracy tests
RMSE = sqrt(residualNorm)/length(residuals)
accuracy = 100*(1 - RMSE)
%RMSE = sqrt((resSquared - meanRS)/length(resSquared))

% Draw the optimal poses in each frame
RtCam3Homog = [RtCam3; 0 0 0 1];
mm = mm.setupview(K3,RtCam3);
mm.draw(mask3, K3, RtCam3, 2);

RtCam2 = RtC3toC2*RtCam3Homog;
RtCam2Homog = [RtCam2; 0 0 0 1];
mm = mm.setupview(K2,RtCam2);
mm.draw(mask2, K2, RtCam2, 2);

RtCam1 = RtC2toC1*RtCam2Homog;
mm = mm.setupview(K1,RtCam1);
mm.draw(mask1, K1, RtCam1, 2);

% Display in subfigures
Hnew = [RtCam3; 0 0 0 1];
Xv = hom_transform(Hnew, Xv2);
%fh = figure;
%subplot(2,2,1);  imagescg(mask3);  title('Cam3');
%subplot(2,2,2);  imagescg(mask2);  title('Cam2');
%subplot(2,2,3);  imagescg(mask1);  title('Cam1');

xvC3 = hom_transform(P3,Xv);  
xvC2 = hom_transform(P2,Xv);
xvC1 = hom_transform(P1,Xv);
%subplot(2,2,1);  hold on;  shC3 = scatter(xvC3(1,:),xvC3(2,:),'r.');
%subplot(2,2,2);  hold on;  shC2 = scatter(xvC2(1,:),xvC2(2,:),'r.');
%subplot(2,2,3);  hold on;  shC1 = scatter(xvC1(1,:),xvC1(2,:),'r.');
return;