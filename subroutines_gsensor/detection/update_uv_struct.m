function [uv_struct, I_orig] = update_uv_struct(uv_struct, image_file, ii, params_G, kernel)
    [uv_struct.line, uv_struct.dist, I_orig] = update_line(image_file, params_G, uv_struct.line, ...
        uv_struct.t, uv_struct.e, uv_struct.n, uv_struct.o, uv_struct.is_opposite, kernel);
    uv_struct.dist_array(ii) = uv_struct.dist;
end