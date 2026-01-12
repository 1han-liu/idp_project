function [n, v, vc] = calc_normal(t, e)
    n = cross(e - t, [0, 0, 1]);
    n = n / norm(n);
    vc = (t + e) / 2;
    v = vc + norm(e - t) * n / 10;
end