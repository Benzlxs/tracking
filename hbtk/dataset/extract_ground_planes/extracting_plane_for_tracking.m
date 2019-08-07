%% Extracting the road plane within point cloud.
% written by xuesong li(xuesong.li@unsw.edu.au) on 29th July 2018
% Code is used to extract road plane and save into txt file.
% reading point cloud -> project into image --> filtering points located in the road--> extract plane -> save into txt file
%% one sentence one branch


function []=main(root_dir)
 
    clear; dbstop error; clc; close all;
    warning off;
    
    if nargin == 0
        root_dir  = '/home/ben/Dataset/KITTI/2011_09_26/2011_09_26_drive_0093_sync';  %% training and testing are strictly separated
    end
    
    ground_height = - 1.1;
    longest_distance       = 70;      % the longest distance we can see
    back_threshold            =  2  ;    % threshold to remove back point
    
    result_dir = [root_dir,'/ground_plane/data/'];
    % mkdir result_dir;
    
    
    velo_dir  = [root_dir, '/reduced_points/data/'];
    assert(exist(velo_dir) == 7,'velody_dir does not exist');
    image_dir = [root_dir, '/image_02/data/'];
    assert(exist(image_dir) == 7,'caliab_dir does not exist');
    
    velos  = dir(fullfile(velo_dir,'*.bin'));   
    images = dir(fullfile(image_dir,'*.png'));
    
    
    if exist(result_dir) ~=7
        mkdir(result_dir);
    end
    
    num_frame = length(velos);
    fst_frame = 1;
    
    for i = fst_frame:1:num_frame
        % read velody data
        velo_file_path = [];
        img_file_path = [];
        check1 =  split(velos(i).name,'.');
        n1 = str2num(cell2mat(check1(1)));
        check3 =  split(images(i).name,'.');
        n3 = str2num(cell2mat(check3(1)));
        assert(n1==n3,'name should be the same')
        
        velo_file_path = [velos(i).folder, '/',  velos(i).name];
        img_file_path  = [images(i).folder, '/', images(i).name];
        velo = filtering_point(velo_file_path);
        % estimate the plane
        idx = []; 
        idx = velo(:,3)< ground_height;
        ground_point = velo(idx,1:3);
        %ground_point = velo(:,1:3);   %% keep the point cloud points filtered by road plane image

        num_pc = size(ground_point , 1);
        dist_plane = [];
        plane_norm = [];
        if num_pc > 1000
            rand_no = floor(num_pc/10);
            if rand_no<=10    % at least 3 points are needed to estimate a plane
                rand_no = num_pc;
            end
            iter_no = 4;
            in_dist_thre = 0.1;
            in_no = floor(num_pc*3/10);      
            rand_no = floor(num_pc/10);
            [ plane_norm  dist_plane] = ransac_plane( ground_point, rand_no, iter_no, in_dist_thre , in_no);
            dist_pc2plane = velo(:,1:3)*plane_norm - dist_plane; % the distance of all points to plane
            idx=[];
            idx = find(dist_pc2plane < in_dist_thre*2);
            ground_point = [];
            ground_point = velo(idx,:);
            velo(idx,:) = [];            
        else
            plane_norm = [-1.980150e-02 ; 7.051739e-03 ; 9.997791e-01] ;  %% setting as the default setting.
            dist_plane   = -1.680367 ;
        end

        results_file = [result_dir, strrep(velos(i).name,'bin','txt')]; % result_name is similar with calibration
        fid = fopen(results_file,'w');
        fprintf(fid,'# Plane\n');
        fprintf(fid, 'Width %d\n',[4]);
        fprintf(fid, 'Height %d\n',[1]);
        fprintf(fid, '%f %f %f %f' ,[-plane_norm(2), -plane_norm(3), plane_norm(1), -dist_plane]);
        fclose(fid);
    end
    

end

function velo = filtering_point(velo_file_path)

    
    fd = fopen(velo_file_path,'rb');
    if fd < 1
        fprintf('No LIDAR files !!!\n');
        keyboard
    else
        velo = fread(fd,[4 inf],'single')';
        fclose(fd);
    end    
    % % segmenting the point cloud into different layers    

end
