function make_line(ax, p1, p2, line_style)
    plot(ax, [p1(1), p2(1)], [p1(2), p2(2)], 'LineStyle', line_style)
end