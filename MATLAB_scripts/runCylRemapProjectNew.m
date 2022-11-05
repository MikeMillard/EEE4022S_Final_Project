% Sample image cam3
fname3 = '../ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam3/baby_cream_1_cam3.bmp';
im3 = imread(fname3);

% Sample image cam2
fname2 = '../ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam2/baby_cream_1_cam2.bmp';
im2 = imread(fname2);

% Sample image cam1
fname1 = '../ResearchProject/Project_Pics/JJ_Baby_Cream/Undistorted/Cam1/baby_cream_1_cam1.bmp';
im1 = imread(fname1);

% Camera specification
K3 = [  1754.27778850637, 0, 534.187667250147;
        0, 1745.68037008362, 423.071381944903;
        0, 0, 1];
K2 = [  1745.49734946345, 0, 503.266990294940;
        0, 1738.96554428750, 376.542049008766;
        0, 0, 1];
K1 = [  1742.30639361572, 0, 501.080240495776;
        0, 1735.47533657380, 383.051931681055;
        0, 0, 1];

% Object specification
objScale = 1.13;
zv = 0:objScale*0.05:objScale*58;  
% Cam 1: x to x, Cam 2: -90 to 30, Cam 3: x to x
thvd = (-90:0.2:30);  
rc = objScale*45.5;  % cylinder radius mm
mm = cylremapm(zv,thvd,rc);

% Object pose in camera frames (from optimisation)
RtCam3 = [  0.999967716430164,0.00111117814900875,0.00795810156791915,-5.44156619002552;
            0.00796799142184448,-0.00919924268934524,-0.999925939780864,36.7927062780815;
            -0.00103788734724179,0.999957068686983,-0.00920779956291096,290.136583357430];
RtCam2 = [  0.530101069668922,0.847801394678797,0.0150216882057526,3.98952975279653;
            0.00930625559702608,0.0118975015534635,-0.999885915024083,41.0535749553453;
            -0.847883393835867,0.530180388771097,-0.00158297817451744,286.202768289943];
RtCam1 = [  -0.409773930649401,0.912095738486766,0.0129108324448967,2.70368960911017;
            0.0190168740433710,0.0226925648012520,-0.999561606907928,40.8433264419527;
            -0.911988861917600,-0.409348764914494,-0.0266440312494236,294.706685843249];

if 0
  addpath('../ResearchProject/MatlabFiles/3d');
  fh = figure;
  P1 = K1*RtCam1;  ch = null(P1);  c1 = ch(1:end-1)/ch(end);
  P2 = K2*RtCam2;  ch = null(P2);  c2 = ch(1:end-1)/ch(end);
  P3 = K3*RtCam3;  ch = null(P3);  c3 = ch(1:end-1)/ch(end);
  hv1 = drawcamera3d(P1,100,[size(im1,2) size(im1,1)]);
  hv2 = drawcamera3d(P2,100,[size(im1,2) size(im1,1)]);
  hv3 = drawcamera3d(P3,100,[size(im1,2) size(im1,1)]);
  th1 = text(c1(1),c1(2),c1(3),'C1');
  th2 = text(c2(1),c2(2),c2(3),'C2');
  th3 = text(c3(1),c3(2),c3(3),'C3');
  axis equal;  axis vis3d;
  axis([-100 300 -300 200 -50 100]);  view(3);
  Xv = mm.Xv;
  hold on;  sh = scatter3(Xv(1,:),Xv(2,:),Xv(3,:),'r.');  hold off;
  xlabel('x');  ylabel('y');  zlabel('z');
end

% Unwrap images
tic
[imr3, va3] = unwrap(mm, im3, K3, RtCam3, 0);  title('Cam3');
[imr2, va2] = unwrap(mm, im2, K2, RtCam2, 0);  title('Cam2');
[imr1, va1] = unwrap(mm, im1, K1, RtCam1, 0);  title('Cam1');
toc
% Flip images
imr3 = flip(imr3, 1); 
imr2 = flip(imr2, 1); 
imr1 = flip(imr1, 1); 

% Display images
%fh = figure;  image(imr3);  title('imr3');
fh = figure;  image(imr2);  title('imr2');
%fh = figure;  image(imr1);  title('imr1');

% % Get visible portion of unwrapped images
% imvis3 = getVisibleUnwrap(imr3, va3);
% imvis2 = getVisibleUnwrap(imr2, va2);
% imvis1 = getVisibleUnwrap(imr1, va1);
% 
% % Display the visible images
% fh = figure(); image(imvis3);
% fh = figure(); image(imvis2);
% fh = figure(); image(imvis1);
return;

function imvis = getVisibleUnwrap(imr, va)
    % Front facing portion of the tub
    dims = size(imr);
    rowIndex = 1;
    colIndex = 1;
    % If the va value is 1, the imr element is visible
    for i=1:dims(1)
        for j=1:dims(2)
            if va(i, j) == 1
                imvis(rowIndex, colIndex, :) = imr(i, j, :);
                colIndex = colIndex + 1;
            end
        end
        colIndex = 1;
        rowIndex = rowIndex + 1;
    end
end
