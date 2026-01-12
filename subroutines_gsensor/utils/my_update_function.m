function output_txt = my_update_function(~,event_obj)
    % event_obj    Object containing event data structure
    % output_txt   Data cursor text
    pos = get(event_obj, 'Position');
    output_txt = {['x: ' num2str(pos(1))], ['y: ' num2str(pos(2))]};
end