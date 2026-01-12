function uv_struct = update_EKF_G(uv_struct, dt_G, resolution, ii)
    [~] = predict(uv_struct.EKF_G);
    measurement_G = [0; 0; 0];
    measurement_G(1) = uv_struct.dist_array(ii) * resolution;
    try
        measurement_G(2) = (uv_struct.dist_array(ii) - uv_struct.dist_array(ii-1)) / dt_G * resolution;
        measurement_G(3) = (uv_struct.dist_array(ii) + uv_struct.dist_array(ii-2) - ...
            2 * uv_struct.dist_array(ii-1)) / dt_G^2 * resolution;
    catch

    end 

    uv_struct.distance_array(ii) = measurement_G(1);
    uv_struct.G_array(ii) = measurement_G(2);

    uv_struct.x_G_array(:, ii) = correct(uv_struct.EKF_G, measurement_G);

    uv_struct.distance_KF_array(ii) = uv_struct.x_G_array(1, ii);
    uv_struct.G_KF_array(ii) = uv_struct.x_G_array(2, ii);
end