function s = calc_intersect(line1_1, line1_2, line2_1, line2_2)
    A = line1_1(1:2)';
    B = line1_2(1:2)';
    C = line2_1(1:2)';
    D = line2_2(1:2)';
    lambda = [B - A, C - D] \ (C - A);
    E = A + lambda(1) * (B - A);
    s = [E', 0];
end


% function s = calc_intersect(line1_1, line1_2, line2_1, line2_2)
%     A = line1_1(1:2)';
%     B = line1_2(1:2)';
%     C = line2_1(1:2)';
%     D = line2_2(1:2)';
%     lambda = [B - A, C - D] \ (C - A);
%     E = A + lambda(1) * (B - A);
%     s = [E', 0];
% end