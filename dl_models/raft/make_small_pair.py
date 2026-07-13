import cv2
from pathlib import Path

in_left = Path("out/rectified_left.png")
in_right = Path("out/rectified_right.png")

out_dir = Path("out_dl/raft_input_small")
out_dir.mkdir(parents=True, exist_ok=True)

left = cv2.imread(str(in_left), cv2.IMREAD_COLOR)
right = cv2.imread(str(in_right), cv2.IMREAD_COLOR)

if left is None or right is None:
    raise FileNotFoundError("Could not read out/rectified_left.png or out/rectified_right.png")

target_width = 640
scale = target_width / left.shape[1]
target_height = int(left.shape[0] * scale)

target_width = (target_width // 32) * 32
target_height = (target_height // 32) * 32

left_small = cv2.resize(left, (target_width, target_height), interpolation=cv2.INTER_AREA)
right_small = cv2.resize(right, (target_width, target_height), interpolation=cv2.INTER_AREA)

cv2.imwrite(str(out_dir / "rectified_left_small.png"), left_small)
cv2.imwrite(str(out_dir / "rectified_right_small.png"), right_small)

print("Original:", left.shape[1], "x", left.shape[0])
print("Small:", target_width, "x", target_height)
print("Saved:", out_dir / "rectified_left_small.png")
print("Saved:", out_dir / "rectified_right_small.png")