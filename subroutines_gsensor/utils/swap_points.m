function [p1, p2] = swap_points(p1, p2)
    tmp = p1;
    p1 = p2;
    p2 = tmp;
end