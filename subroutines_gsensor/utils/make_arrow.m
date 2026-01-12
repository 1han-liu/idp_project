function make_arrow(ax, p1, p2)
    quiver(p1(1), p1(2), p2(1) - p1(1), p2(2) - p1(2), 0)
end