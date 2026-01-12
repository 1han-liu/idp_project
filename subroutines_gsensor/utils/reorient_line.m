function line = reorient_line(line)
    [line.point1, line.point2] = reorient_points(line.point1, line.point2);
end