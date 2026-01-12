% 3 groups according to corner M (M = A, B, C in order)
% 3 angles in each group: 
%   - small adjacent line angle between edge U and middle point W: UMW
%   - large adjacent line angle between edge V and middle point W: VMW
%   - opposite line angle between edge U and edge point V: UMV
function [UMW, VMW, UMV] = get_angles_d(corner)
    angles_d_all = [
        61.60038904,        82.59533203,        112.0661066;
        38.2970794,         69.71231051,        84.1233756;
        53.22106313,        59.96173898,        70.20060129;
        ];
    corners = struct('A', 1, 'B', 2, 'C', 3);
    angles_d = angles_d_all(corners.(corner),:);
    UMW = angles_d(1);
    VMW = angles_d(2);
    UMV = angles_d(3);
end