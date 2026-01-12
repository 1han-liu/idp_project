% Angle UMW: small adjacent line angle
% Angle VMW: large adjacent line angle
% Angle UMV: opposite line angle
% Assumption: M is at the boundary of parper: M(z) = 0
% ---> If M is at center (is_full is true): all other z values are POSITIVE (because z is positive paper-inwards) 
% ---> If M is at side (is_full is false): z are POSITIVE if M points outwards; z are NEGATIVE, if M points inwards.
% 'direction': 'outwards' or 'inwards' or 'extra_inwards'
function [M, W, U, V] = recover_3d(m, w, u, v, corner, is_full, direction)
    [UMW, VMW, UMV] = get_angles_d(corner);
    M = [m(1:2), 0];
    W_z = @(z_w) [w(1:2), z_w];
    U_z = @(z_u) [u(1:2), z_u];
    V_z = @(z_v) [v(1:2), z_v];
    UMN_z = @(z_w, z_u) calc_angle_d(W_z(z_w) - M, U_z(z_u) - M);
    VMN_z = @(z_w, z_v) calc_angle_d(W_z(z_w) - M, V_z(z_v) - M);
    UMV_z = @(z_u, z_v) calc_angle_d(U_z(z_u) - M, V_z(z_v) - M);
    f = @(z) norm([
        UMN_z(z(1), z(2)) - UMW, 
        VMN_z(z(1), z(3)) - VMW, 
        UMV_z(z(2), z(3)) - UMV]);
    z_0 = [100, 100, 100];
    A = [];
    b = [];
    z_min = [0, 0, 0];
    z_max = [Inf Inf Inf];
    if ~is_full
        switch direction
            case 'inwards'
                z_0 = [-100, 100, 100];
                A = [1 -1 0; 1 0 -1];
                b = [0 0];
                z_min = [-Inf 0 0];
                z_max = [0 Inf Inf];
            case 'extra_inwards'
                z_0 = [-100, -100, -100];
                A = [1 -1 0; 1 0 -1];
                b = [0 0];
                z_min = [-Inf -Inf -Inf];
                z_max = [0 0 0];
            case 'outwards'
                z_0 = [100, 100, 100];
                A = [1 -1 0; 1 0 -1];
                b = [0 0];
                z_min = [0 0 0];
                z_max = [Inf Inf Inf];
            case 'in-1-1' % W >= U >= V >= 0
                z_0 = [100, 100, 100];
                A = [-1 1 0; 0 -1 1];
                b = [0 0];
                z_min = [0 0 0];
                z_max = [Inf Inf Inf];
            case 'in-1-2' % W >= V >= U >= 0
                z_0 = [100, 100, 100];
                A = [-1 0 1; 0 1 -1];
                b = [0 0];
                z_min = [0 0 0];
                z_max = [Inf Inf Inf];
            case 'in-2-1' % W >= U >= 0 >= V
                z_0 = [100, 100, -100];
                A = [-1 1 0; 0 -1 1];
                b = [0 0];
                z_min = [0 0 -Inf];
                z_max = [Inf Inf 0];
            case 'in-2-2' % W >= V >= 0 >= U
                z_0 = [100, -100, 100];
                A = [-1 0 1; 0 1 -1];
                b = [0 0];
                z_min = [0 -Inf 0];
                z_max = [Inf 0 Inf];
            case 'in-3-1' % W >= 0 >= U >= V
                z_0 = [100, -100, -100];
                A = [-1 1 0; 0 -1 1];
                b = [0 0];
                z_min = [0 -Inf -Inf];
                z_max = [Inf 0 0];
            case 'in-3-2' % W >= 0 >= V >= U
                z_0 = [100, -100, -100];
                A = [-1 0 1; 0 1 -1];
                b = [0 0];
                z_min = [0 -Inf -Inf];
                z_max = [Inf 0 0];
            case 'in-4-1' % 0 >= W >= U >= V
                z_0 = [-100, -100, -100];
                A = [-1 1 0; 0 -1 1];
                b = [0 0];
                z_min = [-Inf -Inf -Inf];
                z_max = [0 0 0];
            case 'in-4-2' % 0 >= W >= V >= U
                z_0 = [-100, -100, -100];
                A = [-1 0 1; 0 1 -1];
                b = [0 0];
                z_min = [-Inf -Inf -Inf];
                z_max = [0 0 0];
            case 'out-1-1' % W <= U <= V <= 0
                z_0 = [-100, -100, -100];
                A = [1 -1 0; 0 1 -1];
                b = [0 0];
                z_min = [-Inf -Inf -Inf];
                z_max = [0 0 0];
            case 'out-1-2' % W <= V <= U <= 0
                z_0 = [-100, -100, -100];
                A = [1 0 -1; 0 -1 1];
                b = [0 0];
                z_min = [-Inf -Inf -Inf];
                z_max = [0 0 0];
            case 'out-2-1' % W <= U <= 0 <= V
                z_0 = [-100, -100, 100];
                A = [1 -1 0; 0 1 -1];
                b = [0 0];
                z_min = [-Inf -Inf 0];
                z_max = [0 0 Inf];
            case 'out-2-2' % W <= V <= 0 <= U
                z_0 = [-100, 100, -100];
                A = [1 0 -1; 0 -1 1];
                b = [0 0];
                z_min = [-Inf 0 -Inf];
                z_max = [0 Inf 0];
            case 'out-3-1' % W <= 0 <= U <= V
                z_0 = [-100, 100, 100];
                A = [1 -1 0; 0 1 -1];
                b = [0 0];
                z_min = [-Inf 0 0];
                z_max = [0 Inf Inf];
            case 'out-3-2' % W <= 0 <= V <= U
                z_0 = [-100, 100, 100];
                A = [1 0 -1; 0 -1 1];
                b = [0 0];
                z_min = [-Inf 0 0];
                z_max = [0 Inf Inf];
            case 'out-4-1' % 0 <= W <= U <= V
                z_0 = [100, 100, 100];
                A = [1 -1 0; 0 1 -1];
                b = [0 0];
                z_min = [0 0 0];
                z_max = [Inf Inf Inf];
            case 'out-4-2' % 0 <= W <= V <= U
                z_0 = [100, 100, 100];
                A = [1 0 -1; 0 -1 1];
                b = [0 0];
                z_min = [0 0 0];
                z_max = [Inf Inf Inf];
            otherwise
        end
    end
    options = optimoptions('fmincon', 'Display', 'off');
    [z, ~] = fmincon(f, z_0, A, b, [], [], z_min, z_max, [], options);
    W = W_z(z(1));
    U = U_z(z(2));
    V = V_z(z(3));
end