function [line, dist, I_orig] = update_line(image_file, params_G, line, t, e, n, o, is_opposite, kernel)
    old_line = line;
    path = fullfile(image_file.folder, image_file.name);
    warning('off', 'all')
    I_orig = imread(path);
    warning('on', 'all')
    I = I_orig;
    
    I = find_edge_points(I, line.theta, params_G, kernel);
    % I = find_edge_points_2(I, n, line.theta, params_G);
    [I, ~] = calc_masked_image(I, line.point1, line.point2, n, params_G.width, params_G.ratio);



    [Hs,thetas,rhos] = hough(I,'theta', calc_theta_range(line.theta, params_G.delta_theta));
    Hs(rhos < line.rho - params_G.width / params_G.width_divider | rhos > line.rho + params_G.width / params_G.width_divider) = 0;
    peaks = houghpeaks(Hs, params_G.num_peak, 'NHoodSize', [9, 1]);
    lines=houghlines(I, thetas, rhos, peaks, 'FillGap', 5, 'Minlength', 7);
   

    dist2o = inf;
    line_cand = nan;
    for ii = 1:length(lines)
        lines(ii) = reorient_line(lines(ii));
        lines(ii).point1 = [lines(ii).point1, 0];
        lines(ii).point2 = [lines(ii).point2, 0];
        if norm(lines(ii).point1 - lines(ii).point2) < params_G.len_min
            continue;
        end
        dist2o_new = dot(o - line.point1, n) * (is_opposite - 0.5) * 2;
        if dist2o_new < dist2o
            dist2o = dist2o_new;
            line_cand = lines(ii);
        end
    end % for ii
    clear ii
    
    
    if ~isstruct(line_cand)
        line = old_line;
    else
        line = line_cand;
    end

    dist = abs(dot(line.point1 - t, n));
end