function [fig, ia_obj] = get_interactive_figure(I)
    screen_size = get(0, 'ScreenSize');
    fig = figure('Name', 'Get points', 'NumberTitle', 'off', ...
        'Position', screen_size);
    movegui(fig, 'center')
    ax = axes('Parent', fig);
    imshow(I, 'Parent', ax);
    hold(ax, 'on')
    an = annotation(fig, 'textbox', [.075 0.05 .0 .0], 'String', '.', 'FitBoxToText', 'on');
    datacursormode(fig, 'on')
    dcm_obj = datacursormode(fig);
    set(dcm_obj, 'UpdateFcn', @my_update_function)
    ia_obj = struct('an', an, 'dcm_obj', dcm_obj, 'ax', ax);
end