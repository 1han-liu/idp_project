function corner = choose_corner(I)
    corners = {'A', 'B', 'C'};

    screen_size = get(0, 'ScreenSize');
    fig = figure('Name', 'Select Corner', 'NumberTitle', 'off', ...
        'Position', screen_size); 
    movegui(fig, 'center')
    t = tiledlayout(fig, 2, length(corners));

    ax = nexttile(t, 2);
    imshow(I, 'Parent', ax);
    title(ax, 'Raw image');

    for ii = 1:length(corners)
        [~, ~] = draw_image_and_button(fig, t, ii+3, corners{ii});
    end
    sgtitle('Enlarge image by clicking on it', 'FontSize', 14, 'FontWeight', 'bold')
    
    uiwait(fig);
    % --------------------------- NESTED FUNCTIONS ------------------------
    function [ax, button] = draw_image_and_button(fig, t, ii, corner)
        str_choice = strcat('Corner', {' '}, corner);
        I = imread(fullfile('subroutines_gsensor', 'imgs', strcat('corner_', corner,'.jpg')));
        ax = nexttile(t, ii); title(ax, str_choice);
        imshow(I, 'Parent', ax);
        
        handle_I = findobj(ax, 'Type', 'image');
        handle_I.ButtonDownFcn = @(~,~) open_full_size_image(I);

        button = create_button(ax, str_choice, fig, @(~,~) button_callback(fig, corner));
    end

    function button_callback(fig, corner_selected)
        corner = corner_selected;
        uiresume(fig);  % Resume execution to close the window
        close(fig);     % Close the popup window
        disp(['Selected Corner: ' corner_selected]); % Display the selected corner
    end


    function open_full_size_image(I)
        % Open a new figure and display the full-sized image
        fig_full = figure('Name', strcat('Full-sized image'), 'NumberTitle', 'off');
        imshow(I, 'Parent', axes('Parent', fig_full));
    end

% --------------------------- NESTED FUNCTIONS ------------------------
end