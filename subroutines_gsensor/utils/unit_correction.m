start_time = minutes(16) + seconds(08.33);
end_time = minutes(23) + seconds(19.23);
time_difference = minutes(end_time - start_time);
distance_difference = 86; % [micrometer]
delta_t = distance_difference / length(file_numbers);

time_array = (1:length(file_numbers)) * time_difference / length(file_numbers);
distance_array = dist_array * distance_difference / dist_array(end);
distance_KF_array = dist_KF_array * distance_difference / dist_array(end);
growth_rate_array = (distance_array(2:end) - distance_array(1:end-1)) / delta_t;
growth_rate_KF_array = x_array(2,2:end) / delta_t;

save(fullfile("imgs", "real_data.mat"), "time_difference", "distance_difference", "time_array", "distance_array", "distance_KF_array", "growth_rate_array", "growth_rate_KF_array")

h3 = figure;
hold on
plot(time_array, distance_array, 'b*', 'DisplayName', 'measured')
plot(time_array, distance_KF_array, 'b+:', 'DisplayName', 'smoothed')
hold off
legend();
xlabel('t [min]'), ylabel('x [µm]')
saveas(h3, fullfile('imgs', 'dist_unit.png'))

h4 = figure;
hold on
plot(time_array(1:end-1), growth_rate_array, 'b*', 'DisplayName', 'measured')
plot(time_array(1:end-1), growth_rate_KF_array, 'b+:', 'DisplayName', 'smoothed')
hold off
legend();
xlabel('t [min]'), ylabel('G [µm/s]')
saveas(h4, fullfile('imgs', 'G_unit.png'))