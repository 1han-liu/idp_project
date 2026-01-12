function show_3d(I, M, W, U, V)
    hold on;
    imshow(I)
    M_show = M; M_show(3) = -M_show(3);
    W_show = W; W_show(3) = -W_show(3);
    U_show = U; U_show(3) = -U_show(3);
    V_show = V; V_show(3) = -V_show(3);

    % z_min = min([M_show(3), W_show(3), U_show(3), V_show(3)]);
    % 
    % M_show(3) = M_show(3) - z_min;
    % W_show(3) = W_show(3) - z_min;
    % U_show(3) = U_show(3) - z_min;
    % V_show(3) = V_show(3) - z_min;

    mesh.vertices = [M_show; W_show; U_show; V_show];
    mesh.faces = [1 2 3; 1 2 4; 1 3 4; 2 3 4];
    patch(mesh,'FaceVertexCData',[1; 0; 0 ;0],'FaceColor','flat','facealpha',.1);
    hold off
end