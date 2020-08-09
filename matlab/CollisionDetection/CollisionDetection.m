%% Submitted By: Bilal Ahmad Butt
%% Roll No: L15-4208
%%
% You are required to device a solution (in Matlab) for your car
% having back/rear camera; to calculate distance between the car
% and the rear hurdle/wall in real time and intimate the danger
% of back collision with certain critical distance. This appli-
% cation should be available on day as well as evening using
% car backlights. Needless to say, the only input medium is camera
% NOT a distance-measuring sensor.

close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
warning('off', 'Images:initSize:adjustingMag');

%% Break video into frames
videoName = ('VideoName.MOV'); %% <-- Enter video name here.
vid=VideoReader(videoName);

if ~exist(videoName, 'file')
    errorMessage = sprintf('Error: %s does not exist in the search path folders.', videoName);
    uiwait(warndlg(errorMessage));
    return;
end

numFrames = vid.NumberOfFrames;
pickind='jpg';

% for i = 1:numFrames
%     frames = read(vid,i);
%     imwrite(frames,['Image' int2str(i), '.jpg']);
%     im(i)=image(frames);
% end

% Apply Median Filter for smoothening of Image
for i=1:numFrames
    fName = strcat('Image',int2str(i),'.jpg');
    img = imread(fName);
    gray = rgb2gray(img);
    %gray = firstHoldConvoV2(gray);
    %Kaverage = filter2(fspecial('average',3),J)/255;
    Kwiener = wiener2(gray,[7 7]); % wiener applied on original image
    Kmedian = medfilt2(Kwiener); % median applied after wiener
    [Dedges, ~] = edge(Kmedian, 'Sobel');
    imwrite(Kmedian,['combo' int2str(i), '.jpg']);
end

%% Applying Sobel Alogithm to Detect Edges
for i=1:numFrames
    fName = strcat('filtered',int2str(i),'.jpg');
    img = imread(fName);
    %I = rgb2gray(img);
    [Dedges, ~] = edge(img, 'Sobel');
    imwrite(Dedges,['edged' int2str(i), '.jpg']);
    %imshow(g_default);
end
%%
flag = 0;
figure;

%Iterate over all images
for i=1:numFrames
    fName = strcat('combo',int2str(i),'.jpg');
    img = imread(fName);
    [rows, cols] = size(img);
    
    if (flag~= 1)
        for r =1:50
            for c=1:cols
                if(img(r,c) > 0)
                    flag = 1;
                    break;
                end
            end
        end
    end
    
    imshow(img,[]);
    if(flag ==  1)
        text(140,840, 'Alert', 'Color', 'red','FontSize', 32)
    end
    drawnow;
end
