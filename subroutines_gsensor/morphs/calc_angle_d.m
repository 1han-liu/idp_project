function angle_d = calc_angle_d(line1, line2)
    angle_d = acosd(dot(line1, line2) / (norm(line1) * norm(line2)));
end