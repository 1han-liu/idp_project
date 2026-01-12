function [t, p] = reorient_points(t, p)
    if t(2) > p(2)
        [t, p] = swap_points(t, p);
    elseif t(2) == p(2)
        if t(1) > p(1)
            [t, p] = swap_points(t, p);
        end
    end
end