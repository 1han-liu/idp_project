function button_pos = calc_button_position(pos)
    button_width = 0.1;  % Relative width of the button
    button_height = 0.03; % Relative height of the button
    % Center the button horizontally w.r.t. the axes
    button_offset_x = (pos(3) - button_width) / 2;  % Centering offset
    button_offset_y = -0.04;  % Position below the image

    % Calculate the button position
    button_pos = [pos(1) + button_offset_x, pos(2) + button_offset_y, ...
                  button_width, button_height];
end