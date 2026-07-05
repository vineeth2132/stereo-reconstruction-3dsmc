
outPath = "../out/";
gtDepthPath = outPath + "gt_depth_rectified_float.tiff";

D_gt = imread(gtDepthPath);
D_gt = single(D_gt);

% Invalid pixels should already be NaN if C++ loader wrote them that way.
% Still keep this check for safety.
D_gt(~isfinite(D_gt)) = NaN;
D_gt(D_gt <= 0) = NaN;

figure;
h = imagesc(D_gt);
axis image;
colorbar;
impixelinfo;

set(h, "AlphaData", ~isnan(D_gt));
set(gca, "Color", [1 1 1]);
title("Ground Truth Depth Rectified");