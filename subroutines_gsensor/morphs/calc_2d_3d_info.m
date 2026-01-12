function [u_struct, v_struct] = calc_2d_3d_info(M, W, U, V, ...
    u_op_struct, v_op_struct, u_ad_struct, v_ad_struct, is_full)
    
    UV_list = {U, V};
    uv_struct_list = {};
    if is_full % use opposite
        uv_struct_list{1} = u_op_struct;
        uv_struct_list{2} = v_op_struct;
    else % use adjacent
        uv_struct_list{1} = u_ad_struct;
        uv_struct_list{2} = v_ad_struct;
    end


    for kk = 1:2
        n_3d = cross(W - M, UV_list{kk} - M);
        uv_struct_list{kk}.n_3d = n_3d / norm(n_3d);
        uv_struct_list{kk}.cos = abs(dot(uv_struct_list{kk}.n_3d, uv_struct_list{kk}.n)) / ( ...
            norm(uv_struct_list{kk}.n_3d) * norm(uv_struct_list{kk}.n));
        [uv_struct_list{kk}.theta_0, uv_struct_list{kk}.rho_0, uv_struct_list{kk}.is_opposite] = calc_hough_info(uv_struct_list{kk});
    end
    u_struct = uv_struct_list{1};
    v_struct = uv_struct_list{2};
end