clc;
clear;
close all;

outPath = "../out/";

I = imread(outPath + "rectified_left.png");

D_pred = single(imread(outPath + "depth_float.tiff"));
M_pred = imread(outPath + "valid_depth_mask.tiff") > 0;

D_gt = single(imread(outPath + "gt_depth_rectified_float.tiff"));

D_pred(~M_pred) = NaN;
D_pred(~isfinite(D_pred) | D_pred <= 0) = NaN;

D_gt(~isfinite(D_gt) | D_gt <= 0) = NaN;

validPred = isfinite(D_pred) & D_pred > 0;
validGt   = isfinite(D_gt) & D_gt > 0;
common    = validPred & validGt;

gtOnly    = validGt & ~validPred;
predOnly  = validPred & ~validGt;

figure;
imshow(I);
hold on;

[yGt, xGt] = find(gtOnly);
[yPred, xPred] = find(predOnly);
[yCom, xCom] = find(common);

% Downsample for visualization
stepGt = 5;
stepPred = 30;
stepCom = 1;

scatter(xGt(1:stepGt:end), yGt(1:stepGt:end), 1, "r", "filled");
scatter(xPred(1:stepPred:end), yPred(1:stepPred:end), 1, "b", "filled");
scatter(xCom(1:stepCom:end), yCom(1:stepCom:end), 1, "g", "filled");

title("Validity overlap: red=GT only, blue=prediction only, green=both", ...
      "FontSize", 18);

lgd = legend("GT only", "Prediction only", "Both");
lgd.FontSize = 14;

set(gca, "FontSize", 14);

err = abs(D_pred - D_gt);

[y, x] = find(common);
e = err(common);

figure;
imshow(I);
hold on;

scatter(x, y, 1, e, "filled");

axis image;
colorbar;
colormap jet;
title("Absolute depth error on common valid pixels", ...
      "FontSize", 18);

cb = colorbar;
cb.FontSize = 14;

set(gca, "FontSize", 14);

% ---------- Metrics ----------
gtVals = D_gt(common);
predVals = D_pred(common);

signedErr = predVals - gtVals;
absErr = abs(signedErr);

MAE = mean(absErr);
RMSE = sqrt(mean(signedErr.^2));
AbsRel = mean(absErr ./ gtVals);
MedianAE = median(absErr);

coverageOverGt = nnz(common) / nnz(validGt);
coverageOverPred = nnz(common) / nnz(validPred);

fprintf("\n===== Depth Evaluation Metrics =====\n");
fprintf("Valid GT pixels:          %d\n", nnz(validGt));
fprintf("Valid prediction pixels:  %d\n", nnz(validPred));
fprintf("Common valid pixels:      %d\n", nnz(common));
fprintf("Coverage over GT:         %.2f %%\n", 100 * coverageOverGt);
fprintf("Coverage over prediction: %.2f %%\n", 100 * coverageOverPred);
fprintf("MAE:                      %.4f\n", MAE);
fprintf("RMSE:                     %.4f\n", RMSE);
fprintf("AbsRel:                   %.4f\n", AbsRel);
fprintf("Median AE:                %.4f\n", MedianAE);

fprintf("Error p50:                %.4f\n", prctile(absErr, 50));
fprintf("Error p90:                %.4f\n", prctile(absErr, 90));
fprintf("Error p95:                %.4f\n", prctile(absErr, 95));