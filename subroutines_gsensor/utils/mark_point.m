function p = mark_point(info_str, ia_obj, color_marker)
    info_struct = [];
    while ~isfield(info_struct, 'Position')
        ia_obj.an.String = info_str;
        disp(ia_obj.an.String);
        pause
        info_struct = getCursorInfo(ia_obj.dcm_obj);
    end
    p = info_struct.Position;
    p = [p, 0];
    scatter(ia_obj.ax,p(1), p(2), color_marker)
end