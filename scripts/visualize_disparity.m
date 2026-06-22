outPath = "../out/";
dispPath = outPath + "disparity_float.tiff";
maskPath = outPath + "valid_disparity_mask.tiff";

D = imread(dispPath);
M = imread(maskPath) > 0;

D(~M) = NaN;

h = imagesc(D);
axis image;
colorbar;
impixelinfo;

set(h, 'AlphaData', ~isnan(D));
set(gca, 'Color', [1 1 1]);   % show NaN as white
