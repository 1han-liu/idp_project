function is_full = choose_is_full(I)
    screen_size = get(0, 'ScreenSize');
    fig_orig = figure('Name', 'Select Corner', 'NumberTitle', 'off', ...
        'Position', screen_size); 
    movegui(fig_orig, 'center')
    ax = axes('Parent', fig_orig);
    imshow(I, 'Parent', ax);
    t = tiledlayout(fig_orig, 2, 2);

    [~, ~] = draw_image_and_button(fig_orig, I, t, 3, true);
    [~, ~] = draw_image_and_button(fig_orig, I, t, 4, false);

    uiwait(fig_orig);

    function [ax, button] = draw_image_and_button(fig, I, t, ii, is_full)
        if is_full
            full_str = 'Yes';
        else
            full_str = 'No';
        end
        str_choice = strcat('Is full: ', {' '}, full_str);
        ax = nexttile(t, ii); title(ax, str_choice);
        imshow(I, 'Parent', ax);

        button = create_button(ax, str_choice, fig, @(~,~) button_callback(fig, is_full));
    end

    function button_callback(fig, is_full_selected)
        if is_full_selected
            full_str = 'Yes';
        else
            full_str = 'No';
        end
        is_full = is_full_selected;
        uiresume(fig);  % Resume execution to close the window
        close(fig);     % Close the popup window
        disp(['Is full? ' full_str]); % Display the selected corner
    end
end