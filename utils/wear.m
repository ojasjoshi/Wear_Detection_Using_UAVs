clc
clear


folder_dir = '11';
file_extension = '*.jpg';

full_name = fullfile(folder_dir,file_extension);
img_files = dir(full_name);
nImgs = length(img_files);

rects = zeros(nImgs,4);
conts = zeros(10000,2*length(img_files));
fig = figure;
start_img = 1;
end_img = 10;

nImg = length(img_files);
contour_counter = 1;

for i = start_img:end_img
    im_loc = fullfile(folder_dir,img_files(i).name);
    img = imread(im_loc);
    ax = gca;
    imshow(img,'Parent',ax);
    hold(ax,'on')
    start_val = 1;
    while(1)
        if(start_val == 1)
            exit_val = 0;
        else
            k = waitforbuttonpress;
            exit_val = double(get(gcf,'CurrentCharacter'))-28;
        end
        
        if(exit_val == 0)
            hFH = imfreehand('Closed',false);
            xy=hFH.getPosition();
            plot(xy(:,1),xy(:,2),'r*','LineWidth',3)
            nBox = size(xy,1);
            
            conts(1,2*contour_counter-1) = i;
            conts(1,2*contour_counter) = nBox;
            conts(2:nBox+1,[2*contour_counter-1,2*contour_counter]) = xy;
            start_val = 0;
            contour_counter = contour_counter+1;
        else
            break;
        end
    end
    
    pause(0.3)
    hold(ax,'off')
    
end
close all
save(['all_rects_',num2str(start_img),'_',num2str(end_img)],'conts','img_files')
function val_returned = myfun(src,event)
if(strcmp(event.Key,'return'))
    val_returned = 1;
end
end

%     rect = getrect;
%     rectangle('Position',rect, 'EdgeColor','r','LineWidth',3)
%     rects(i,:) = [rect([1,2]),rect([3,4])+  rect([1,2])];