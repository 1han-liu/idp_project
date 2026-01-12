function [I, mask] = calc_masked_image(I, t, e, n, width, ratio)
    up = t + n * width;
    dp = t - n * width;
    s = e - t;
    rp = e + (e - t) * ratio;
    lp = t - (e - t) * ratio;
    size_I = size(I);
    [x, y] = meshgrid(1:size_I(2), 1:size_I(1));
    mask1 = ((up(1) - x) .* n(1) + (up(2) - y) .* n(2)) .* ( ...
        (dp(1) - x) .* n(1) + (dp(2) - y) .* n(2)) < 0;
    % mask2 = ((rp(1) - x) .* s(1) + (rp(2) - y) .* s(2)) .* ( ...
    %     (lp(1) - x) .* s(1) + (lp(2) - y) .* s(2)) < 0;
    % mask = mask1 .* mask2;
    mask = mask1;
    try
        I = I .* mask;
    catch
        I(~mask(:, :, ones(1, size(I, 3)))) = 0;
    end
end