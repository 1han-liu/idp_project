function h = update_figure(u_struct, v_struct, I, file)
    h = figure('visible', 'off'); 
    movegui(h, 'center')
    ax_G = axes('Parent', h, 'Position', [0, 0, 1, 1]);
    imshow(I, 'Parent', ax_G), hold(ax_G, 'on')

    color_base = 255;   
    u_color = [0,250,154] / color_base; % medium spring green
    v_color = [210,105,30] / color_base; % chocoloate
    
    plot([u_struct.t(1), u_struct.e(1)], [u_struct.t(2), u_struct.e(2)], ...
        'Color', u_color, 'LineStyle', ':', 'Parent', ax_G, 'LineWidth', 1, 'DisplayName', 'edge u (orig)')
    plot([v_struct.t(1), v_struct.e(1)], [v_struct.t(2), v_struct.e(2)], ...
        'Color', v_color, 'LineStyle', ':', 'Parent', ax_G, 'LineWidth', 1, 'DisplayName', 'edge u (current)')

    plot([u_struct.line.point1(1), u_struct.line.point2(1)], [u_struct.line.point1(2), u_struct.line.point2(2)], ...
        'Color', u_color, 'LineStyle', '-', 'Parent', ax_G, 'LineWidth', 2, 'DisplayName', 'edge v (orig)')
    plot([v_struct.line.point1(1), v_struct.line.point2(1)], [v_struct.line.point1(2), v_struct.line.point2(2)], ...
        'Color', v_color, 'LineStyle', '-', 'Parent', ax_G, 'LineWidth', 2, 'DisplayName', 'edge v (current)')
    hold(ax_G, 'off')
    if ~exist(fullfile('..', 'gsensor_data'), 'dir')
        mkdir(fullfile('..', 'gsensor_data'))
    end
    if ~exist(fullfile('..', 'gsensor_data', 'images'), 'dir')
        mkdir(fullfile('..', 'gsensor_data', 'images'))
    end
    saveas(h, fullfile('..', 'gsensor_data', 'images', strrep(file.name, '.PNG', '.jpg')))
end