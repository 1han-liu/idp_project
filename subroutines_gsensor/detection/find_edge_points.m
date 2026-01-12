function I = find_edge_points(I, theta, params_G, kernel)
    I = rgb2gray(I);

    T = max(1, graythresh(I) * 1.2);
    I(I > T * 255) = T * 255;
    I = medfilt2(I, [7, 7], 'symmetric');
    I = medfilt2(I, [100, 100], 'symmetric');

    I = imbinarize(I,'adaptive','ForegroundPolarity','dark');
    I = imclose(~I, strel('disk', 10));
    I = imopen(I, strel('disk', 10));

    I_wo_kernel = I;
   
    I_kernel = calc_kernel_mask(I, kernel);
    I(I_kernel) = 1;
    I = imclose(I, strel('disk', 100));
    stat = regionprops(I, 'Area');
    area = max([stat.Area]);
    I = bwareaopen(I, area);

    I = I .* I_wo_kernel;
    I = edge(I, 'nothinning');
    I = imdilate(I, strel('disk', 3));
end