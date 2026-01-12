function [theta, rho, is_opposite] = calc_hough_info(uv_struct)
    theta = acosd(dot(uv_struct.n,[1 0 0])) * sign(uv_struct.n(2));
    rho = dot(uv_struct.n, uv_struct.t);
    is_opposite = dot(uv_struct.o - uv_struct.t, uv_struct.n) < 0;
end