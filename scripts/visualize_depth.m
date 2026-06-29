outPath = "../out/";
depthPath = outPath + "depth_float.tiff";
maskPath = outPath + "valid_depth_mask.tiff";

D = imread(depthPath);
M = imread(maskPath) > 0;

D(~M) = NaN;

h = imagesc(D);
axis image;
colorbar;
impixelinfo;

set(h, 'AlphaData', ~isnan(D));
set(gca, 'Color', [1 1 1]);   % show NaN as white
