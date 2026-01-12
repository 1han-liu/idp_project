function theta_range = calc_theta_range(theta, delta_theta)
    theta_range = linspace(theta-delta_theta, theta+delta_theta, 20);%+-20deg fot measurment inacc
    theta_range = sort(mod(theta_range + 90,180) - 90, 'ascend');
end