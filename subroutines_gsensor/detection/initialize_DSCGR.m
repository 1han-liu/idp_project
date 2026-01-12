function [uv_struct_list, kernel] = initialize_DSCGR(image_file)
    path = fullfile(image_file.folder, image_file.name);
    warning('off', 'all') 
    I = imread(path);
    warning('on', 'all')

    is_full = choose_is_full(I);
    [fig_2d, ax_2d, m, w, u, v, u_op_struct, v_op_struct, u_ad_struct, v_ad_struct, kernel] = get_points(I, is_full); 
    corner = choose_corner(I);
    [fig_3d, ax_3d, M, W, U, V] = recover_3d_all(I, m, w, u, v, corner, is_full); %#ok<*ASGLU> 
    uv_struct_list = {};
    [uv_struct_list{1}, uv_struct_list{2}] = calc_2d_3d_info(M, W, U, V, ...
        u_op_struct, v_op_struct, u_ad_struct, v_ad_struct, is_full);
    if ~exist(fullfile('..', 'gsensor_data'), 'dir')
        mkdir(fullfile('..', 'gsensor_data'))
    end
    save(fullfile('..', 'gsensor_data', "DSCGR_info.mat"))
end