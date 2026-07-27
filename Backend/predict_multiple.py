import io
import base64
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

from predict_single import (
    COLORS,
    YOLO_MULTI_IMGSZ,
    ensure_initialized,
    encode_image,
    is_animal_image,
    clip_predict_one,
    crop_with_padding,
    get_yolo,
    ANIMAL_KEYWORDS,
    _box_area,
    _compute_iou,
    animal_names,
    text_matrix,
)

def _get_font(size: int = 20):
    for path in [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _text_size(draw, text: str, font, fallback_char_w=12, fallback_h=24):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return len(text) * fallback_char_w, fallback_h

def detect_multiple_animals(original_img: Image.Image) -> List[dict]:
    W, H = original_img.size
    img_area = W * H
    yolo = get_yolo()
    detections = []

    for imgsz in YOLO_MULTI_IMGSZ:
        results = yolo(
            original_img,
            conf=0.05,
            iou=0.45,
            imgsz=imgsz,
            augment=True,
            verbose=False,
        )
        result = results[0]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        class_names = result.names

        for cls_id, conf_score, box in zip(class_ids, confidences, xyxy):
            cname = class_names.get(int(cls_id), str(cls_id))
            if not any(kw in cname.lower() for kw in ANIMAL_KEYWORDS):
                continue
            x1, y1, x2, y2 = box.tolist()
            clipped_box = [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]
            b_area = _box_area(clipped_box)
            if b_area > 0.85 * img_area or b_area < 0.0004 * img_area:
                continue
            detections.append({
                "box": clipped_box,
                "yolo_class": cname,
                "confidence": float(conf_score),
                "imgsz": imgsz,
            })

    # Simple deduplication
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in ordered:
        merged = False
        for existing in kept:
            if _compute_iou(det["box"], existing["box"]) > 0.55:
                merged = True
                break
        if not merged:
            kept.append(det)

    kept = sorted(kept, key=lambda d: (d["box"][1], d["box"][0]))
    for i, det in enumerate(kept, start=1):
        det["label"] = i
    return kept

def handle_predict_multiple(original_img: Image.Image) -> Tuple[dict, int]:
    """
    Main handler for multiple-animal prediction endpoint.
    """
    ensure_initialized()
    kept_detections = detect_multiple_animals(original_img)

    if not kept_detections:
        img_embedding = encode_image(original_img)
        is_anim, animal_score, reason = is_animal_image(img_embedding)
        if not is_anim:
            return {
                "error": "Not an animal image",
                "detail": reason,
                "animal_score": round(animal_score * 100, 2),
            }, 400
        else:
            return {
                "error": "No animals detected",
                "detail": "Object detection did not find any animal objects in this image.",
            }, 400

    font = _get_font(20)
    final_detections = []
    annotated_img = original_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    for i, det in enumerate(kept_detections, start=1):
        x1, y1, x2, y2 = det["box"]
        color = COLORS[(i - 1) % len(COLORS)]
        crop = crop_with_padding(original_img, det["box"], padding_ratio=0.05)
        crop_embedding = encode_image(crop)
        # Wider context crop for ensemble
        context_crop = crop_with_padding(original_img, det["box"], padding_ratio=0.30)
        context_embedding = encode_image(context_crop)
        is_anim, animal_score, reason = is_animal_image(crop_embedding)

        if not is_anim:
            name, accuracy = None, 0.0
            label_name = "Not an animal"
        else:
            name, accuracy = clip_predict_one(
                crop_embedding,
                yolo_class=det.get("yolo_class"),
                yolo_confidence=det.get("confidence", 0.0),
                context_embedding=context_embedding,
            )
            label_name = name or det.get("yolo_class", "Animal")

        # Draw bounding box on annotated image
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tag = f"#{i} {label_name}"
        tw, th = _text_size(draw, tag, font)
        ty = max(0, y1 - th - 8)
        draw.rectangle([x1, ty, x1 + tw + 12, ty + th + 8], fill=color)
        draw.text((x1 + 6, ty + 4), tag, fill="#FFFFFF", font=font)

        # Convert crop to base64
        crop_buffer = io.BytesIO()
        crop.save(crop_buffer, format="JPEG", quality=90)
        crop_base64 = base64.b64encode(crop_buffer.getvalue()).decode("utf-8")

        final_detections.append({
            "box": [x1, y1, x2, y2],
            "yolo_class": det.get("yolo_class"),
            "detection_confidence": round(det.get("confidence", 0.0) * 100.0, 2),
            "detection_imgsz": det.get("imgsz"),
            "name": label_name,
            "accuracy": accuracy,
            "is_animal": is_anim,
            "crop_image": f"data:image/jpeg;base64,{crop_base64}",
        })

    ann_buffer = io.BytesIO()
    annotated_img.save(ann_buffer, format="JPEG", quality=90)
    ann_base64 = base64.b64encode(ann_buffer.getvalue()).decode("utf-8")

    return {
        "annotated_image": f"data:image/jpeg;base64,{ann_base64}",
        "detections": final_detections,
    }, 200

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_img = Image.open(sys.argv[1]).convert("RGB")
        res, code = handle_predict_multiple(test_img)
        print("Multiple Result Detections:", len(res.get("detections", [])), f"({code})")