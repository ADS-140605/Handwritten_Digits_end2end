import argparse
import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

try:
    from .model import MNISTCNN
except ImportError:
    from model import MNISTCNN


def otsu_threshold(gray):
	hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
	total = gray.size
	sum_total = np.dot(np.arange(256), hist)
	weight_bg = 0.0
	sum_bg = 0.0
	best_var = -1.0
	best_t = 127

	for t in range(256):
		weight_bg += hist[t]
		if weight_bg == 0:
			continue
		weight_fg = total - weight_bg
		if weight_fg == 0:
			break
		sum_bg += t * hist[t]
		mean_bg = sum_bg / weight_bg
		mean_fg = (sum_total - sum_bg) / weight_fg
		between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
		if between > best_var:
			best_var = between
			best_t = t

	return best_t


def connected_components(mask, min_area=25):
	h, w = mask.shape
	visited = np.zeros_like(mask, dtype=bool)
	boxes = []

	for y in range(h):
		for x in range(w):
			if not mask[y, x] or visited[y, x]:
				continue

			queue = deque([(y, x)])
			visited[y, x] = True
			pixels = []
			min_y, max_y = y, y
			min_x, max_x = x, x

			while queue:
				cy, cx = queue.popleft()
				pixels.append((cy, cx))
				if cy < min_y:
					min_y = cy
				if cy > max_y:
					max_y = cy
				if cx < min_x:
					min_x = cx
				if cx > max_x:
					max_x = cx

				for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
					if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
						visited[ny, nx] = True
						queue.append((ny, nx))

			if len(pixels) >= min_area:
				boxes.append((min_x, min_y, max_x + 1, max_y + 1))

	return boxes


def sort_boxes_reading_order(boxes):
	if not boxes:
		return []

	heights = [b[3] - b[1] for b in boxes]
	median_h = float(np.median(np.array(heights, dtype=np.float64)))
	line_tol = max(8.0, 0.7 * median_h)

	boxes_sorted = sorted(boxes, key=lambda b: ((b[1] + b[3]) * 0.5, b[0]))
	lines = []

	for box in boxes_sorted:
		cy = (box[1] + box[3]) * 0.5
		placed = False
		for line in lines:
			if abs(cy - line["y_center"]) <= line_tol:
				line["boxes"].append(box)
				line["y_center"] = np.mean([(b[1] + b[3]) * 0.5 for b in line["boxes"]])
				placed = True
				break
		if not placed:
			lines.append({"y_center": cy, "boxes": [box]})

	lines.sort(key=lambda l: l["y_center"])
	for line in lines:
		line["boxes"].sort(key=lambda b: b[0])

	return lines


def preprocess_digit(gray, box, pad=4):
	x0, y0, x1, y1 = box
	crop = gray[y0:y1, x0:x1]
	h, w = crop.shape
	s = max(h, w) + 2 * pad
	canvas = np.zeros((s, s), dtype=np.uint8)
	y_start = (s - h) // 2
	x_start = (s - w) // 2
	canvas[y_start : y_start + h, x_start : x_start + w] = crop

	img = Image.fromarray(canvas, mode="L")
	img = img.resize((28, 28), Image.Resampling.BICUBIC)
	return img


def build_foreground_mask(gray):
	threshold = otsu_threshold(gray)
	mask_dark = gray < threshold
	mask_light = gray > threshold

	dark_ratio = mask_dark.mean()
	light_ratio = mask_light.mean()

	def score(ratio):
		if ratio < 0.005 or ratio > 0.8:
			return 10.0
		return abs(ratio - 0.15)

	return mask_dark if score(dark_ratio) <= score(light_ratio) else mask_light


def predict_from_image(image_path, model_path, output_dir, device, min_area=30):
	os.makedirs(output_dir, exist_ok=True)

	img = Image.open(image_path).convert("L")
	gray = np.array(img)
	mask = build_foreground_mask(gray)
	boxes = connected_components(mask, min_area=min_area)
	lines = sort_boxes_reading_order(boxes)

	if not lines:
		raise RuntimeError("No digit-like components found. Try a cleaner image with stronger contrast.")

	model = MNISTCNN().to(device)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model.eval()

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
		]
	)

	annotated = img.convert("RGB")
	draw = ImageDraw.Draw(annotated)
	pred_lines = []

	with torch.no_grad():
		for line in lines:
			line_digits = []
			for box in line["boxes"]:
				x0, y0, x1, y1 = box
				digit_patch = (mask[y0:y1, x0:x1].astype(np.uint8)) * 255
				patch_img = preprocess_digit(digit_patch, (0, 0, x1 - x0, y1 - y0))
				input_tensor = transform(patch_img).unsqueeze(0).to(device)
				logits = model(input_tensor)
				pred = int(logits.argmax(dim=1).item())
				line_digits.append(str(pred))

				draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
				draw.text((x0, max(0, y0 - 12)), str(pred), fill=(0, 255, 0))

			pred_lines.append("".join(line_digits))

	result_text = "\n".join(pred_lines)
	text_path = os.path.join(output_dir, "predicted_text.txt")
	with open(text_path, "w", encoding="utf-8") as f:
		f.write(result_text + "\n")

	viz_path = os.path.join(output_dir, "predicted_overlay.jpg")
	annotated.save(viz_path)

	print("Predicted text:")
	print(result_text)
	print(f"Saved text to {text_path}")
	print(f"Saved overlay to {viz_path}")



def main():

    parser = argparse.ArgumentParser(
        description="Predict handwritten digits from a full image"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to handwritten digit image (jpg/png)"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(
            "..",
            "backend",
            "storage",
            "models",
            "original",
            "mnist_cnn.pt",
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=30,
        help="Minimum component area (lower = more sensitive)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(
            f"Input image not found: {args.image}"
        )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available. Using CPU.")

    predict_from_image(
        args.image,
        args.model_path,
        args.output_dir,
        device,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()