function side_struct = get_side(ia_obj, foot, angle, suffix, p_text_func_general)
    p_text_add = strcat(angle , {' '}, 'angle', {' at '}, suffix , {' '}, 'side');
    p_text_func = @(order, p_str) p_text_func_general(order, p_str, p_text_add);

    if ~strcmp(foot, 'w')
        side_str_with_suffix = strcat(foot, '\_', suffix(1:2));
        t_str = strcat('t\_', side_str_with_suffix);
        e_str = strcat('e\_', side_str_with_suffix);
        n_str = strcat('n\_', side_str_with_suffix);
        o_str = strcat('o\_', side_str_with_suffix);
    
        t = mark_point(p_text_func('first', t_str), ia_obj, 'b*');
        e = mark_point(p_text_func('second', e_str), ia_obj, 'b*');
        [t, e] = reorient_points(t, e);
        annotate_point(ia_obj.ax, t, t_str)
        annotate_point(ia_obj.ax, e, e_str)
        [n, v, vc] = calc_normal(t, e);
        annotate_point(ia_obj.ax, (v + vc) / 2, n_str)
        make_arrow(ia_obj.ax, t, e)
        make_arrow(ia_obj.ax, vc, v)
        o = mark_point(p_text_func('outer', o_str), ia_obj, 'go');
        annotate_point(ia_obj.ax, o, o_str)
    
        side_struct = struct('t', t, 'e', e, 'n', n, 'v', v, 'vc', vc, 'o', o, ...
            'foot', foot, 'suffix', suffix);
    else
        w_str = 'w'; 
        w = mark_point(p_text_func('middle', w_str), ia_obj, 'rx');
        annotate_point(ia_obj.ax, w, w_str)
        side_struct = struct('w', w);
    end

end