function [fig_3d_all, ax_3d_all, choice, M, W, U, V] = recover_3d_all(I, m, w, u, v, corner, is_full)
    screen_size = get(0, 'ScreenSize');
    fig_3d_all = figure('Name', 'Select Corner', 'NumberTitle', 'off', ...
        'Position', screen_size); 
    movegui(fig_3d_all, 'center')
    ax_3d_all = axes('Parent', fig_3d_all);
    t = tiledlayout(fig_3d_all, 4, 4);
    choice = 0;
    M = m; W = w; U = u; V = v;
    if is_full
        ii = 2;
        direction = 'outwards';
        str_choice = strcat('Corner points', {' '}, direction);
        str_choice = str_choice{1};
        [~, ~] = draw_figure_and_button(...
            m, w, u, v, corner, is_full, direction, fig_3d_all, t, ii, ii, str_choice);
    else
        directions = {'in-1-1', 'in-1-2', 'in-2-1', 'in-2-2', 'in-3-1', 'in-3-2', 'in-4-1', 'in-4-2', ...
            'out-1-1', 'out-1-2', 'out-2-1', 'out-2-2', 'out-3-1', 'out-3-2', 'out-4-1', 'out-4-2'};
        for ii = 1 : length(directions)
            direction = directions{ii};
            str_choice = strcat('Corner points', {' '}, direction);
            str_choice = str_choice{1};
            [~, ~] = draw_figure_and_button(...
                m, w, u, v, corner, is_full, direction, fig_3d_all, t, ii, ii, str_choice);
        end
        
    end
    sgtitle('All possible solutions', 'FontSize', 14, 'FontWeight', 'bold')
    uiwait(fig_3d_all);

    

    function [ax, button] = draw_figure_and_button(m, w, u, v, corner, is_full, direction, fig, t, ii, choice, str_choice)
        ax = nexttile(t, ii); title(ax, str_choice)
        [M, W, U, V] = recover_3d(m, w, u, v, corner, is_full, direction);
        show_3d(I, M, W, U, V);
        button = create_button(ax, str_choice, fig, ...
            @(~,~) button_callback(fig, choice, str_choice, M, W, U, V));
    end

    function button_callback(fig, choice_selected, str_choice, M_selected, W_selected, U_selected, V_selected)
        M = M_selected; W = W_selected; U = U_selected; V = V_selected;
        choice = choice_selected;
        uiresume(fig);  % Resume execution to close the window
        close(fig);     % Close the popup window
        fprintf('Choice: %s\n', str_choice); % Display the selected choice
    end

end