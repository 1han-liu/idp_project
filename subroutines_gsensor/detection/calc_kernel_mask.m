function kernel_mask = calc_kernel_mask(I, kernel)
    size_I = size(I);
    [x, y] = meshgrid(1:size_I(2), 1:size_I(1));
    kernel_mask = true(size(I));
    num_corners = length(kernel.k_c_cell);
    for ii = 1:num_corners
        p1 = kernel.k_c_cell{ii};
        p2 = kernel.k_c_cell{mod(ii, num_corners) + 1};
        tau = p2 - p1;
        n = [tau(2), -tau(1), 0];
        o = kernel.k_o_cell{ii};
        sign = (p1(1) - o(1)) .* n(1) + (p1(2) - o(2)) .* n(2);
        mask_tmp = sign * ((p1(1) - x) .* n(1) + (p1(2) - y) .* n(2)) < 0;
        kernel_mask = kernel_mask .* mask_tmp;
    end
    kernel_mask = kernel_mask > 0;
end