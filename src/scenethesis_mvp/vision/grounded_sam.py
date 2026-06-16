from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import DetectionSpec, SegmentationResult
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class GroundedSAMConfig:
    grounding_dino_config: Path
    grounding_dino_checkpoint: Path
    sam_checkpoint: Path
    sam_model_type: str = "vit_h"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    retry_box_threshold: float = 0.20
    retry_text_threshold: float = 0.16
    device: str = "cuda"
    min_mask_pixels: int = 128


class GroundedSAMSegmenter:
    def __init__(self, config: GroundedSAMConfig):
        self.config = config

    def segment(
        self,
        image_path: str | Path,
        scene: SceneSpec,
        out_dir: str | Path,
    ) -> SegmentationResult:
        self._validate_files()
        target_dir = Path(out_dir)
        masks_dir = target_dir / "masks"
        crops_dir = target_dir / "crops"
        masks_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            from groundingdino.util.inference import annotate, load_image, load_model, predict
            from segment_anything import SamPredictor, sam_model_registry
            from torchvision.ops import box_convert
        except Exception as exc:
            raise RuntimeError(f"Grounded-SAM dependencies are not installed correctly: {exc}") from exc

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM requested CUDA, but torch.cuda.is_available() is false.")

        object_prompt_map = build_object_prompt_candidates(scene)
        if not object_prompt_map:
            raise RuntimeError("Grounded-SAM requires at least one planned object prompt.")

        model = load_model(
            str(self.config.grounding_dino_config),
            str(self.config.grounding_dino_checkpoint),
            device=self.config.device,
        )
        image_source, image_tensor = load_image(str(image_path))
        image_height, image_width = image_source.shape[:2]
        selected_boxes_xyxy = []
        selected_logits = []
        selected_phrases = []
        selected_object_ids = []
        selected_box_arrays = []

        for obj in scene.objects:
            prompts = object_prompt_map[obj.id]
            caption = " . ".join(prompts) + " ."
            boxes, logits, phrases = [], [], []
            for box_threshold, text_threshold in [
                (self.config.box_threshold, self.config.text_threshold),
                (self.config.retry_box_threshold, self.config.retry_text_threshold),
            ]:
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.config.device,
                )
                if len(boxes) > 0:
                    break
            if len(boxes) == 0:
                continue
            boxes_xyxy = box_convert(
                boxes=boxes * torch.tensor([image_width, image_height, image_width, image_height], device=boxes.device),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            selected_index = select_non_overlapping_detection(boxes_xyxy, logits, selected_box_arrays)
            if selected_index is None:
                continue
            box_xyxy = boxes_xyxy[selected_index]
            selected_boxes_xyxy.append(box_xyxy)
            selected_logits.append(logits[selected_index])
            selected_phrases.append(str(phrases[selected_index]))
            selected_object_ids.append(obj.id)
            selected_box_arrays.append(box_xyxy.detach().cpu().numpy().astype(float))

        if not selected_boxes_xyxy:
            all_prompts = [" / ".join(prompts) for prompts in object_prompt_map.values()]
            result = SegmentationResult(
                image_path=str(image_path),
                image_width=image_width,
                image_height=image_height,
                detections=[],
                missing_object_ids=[obj.id for obj in scene.objects],
            )
            write_json(target_dir / "segmentation.json", result)
            raise RuntimeError("GroundingDINO found no objects for planned object prompts: " + "; ".join(all_prompts))

        sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(self.config.sam_checkpoint))
        sam.to(device=self.config.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image_source)
        boxes_xyxy_tensor = torch.stack(selected_boxes_xyxy).to(self.config.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy_tensor, image_source.shape[:2]).to(self.config.device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        detections = []
        assigned_object_ids: set[str] = set()
        pil_image = Image.fromarray(image_source)
        for index, object_id in enumerate(selected_object_ids):
            mask_np = masks[index, 0].detach().cpu().numpy().astype(bool)
            mask_area = int(mask_np.sum())
            if mask_area < self.config.min_mask_pixels:
                continue
            box = selected_boxes_xyxy[index].detach().cpu().numpy().astype(float)
            assigned_object_ids.add(object_id)
            mask_path = masks_dir / f"{object_id or 'unmatched'}_{index:03d}.png"
            crop_path = crops_dir / f"{object_id or 'unmatched'}_{index:03d}.png"
            Image.fromarray((mask_np.astype(np.uint8) * 255)).save(mask_path)
            crop_with_mask(pil_image, mask_np, box).save(crop_path)
            detections.append(
                DetectionSpec(
                    object_id=object_id,
                    phrase=selected_phrases[index],
                    score=float(selected_logits[index].detach().cpu().item()),
                    box_xyxy=[round(float(value), 3) for value in box.tolist()],
                    mask_path=str(mask_path),
                    crop_path=str(crop_path),
                    mask_area=mask_area,
                )
            )

        missing_object_ids = [obj.id for obj in scene.objects if obj.id not in assigned_object_ids]
        overlay_path = target_dir / "segmentation_overlay.png"
        draw_segmentation_overlay(Image.fromarray(image_source), detections).save(overlay_path)
        result = SegmentationResult(
            image_path=str(image_path),
            image_width=image_width,
            image_height=image_height,
            detections=detections,
            missing_object_ids=missing_object_ids,
            overlay_path=str(overlay_path),
        )
        write_json(target_dir / "segmentation.json", result)
        if missing_object_ids:
            raise RuntimeError(
                "Grounded-SAM did not produce masks for required objects: "
                + ", ".join(missing_object_ids)
                + ". Faithful mode cannot continue without segmentation."
            )
        return result

    def _validate_files(self) -> None:
        required = {
            "GroundingDINO config": self.config.grounding_dino_config,
            "GroundingDINO checkpoint": self.config.grounding_dino_checkpoint,
            "SAM checkpoint": self.config.sam_checkpoint,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
        if missing:
            raise RuntimeError("Missing Grounded-SAM files:\n- " + "\n- ".join(missing))


def build_object_prompts(scene: SceneSpec) -> list[str]:
    return [prompts[0] for prompts in build_object_prompt_candidates(scene).values()]


def build_object_prompt_candidates(scene: SceneSpec) -> dict[str, list[str]]:
    prompt_map = {}
    for obj in scene.objects:
        terms = [
            (obj.name or obj.category).replace("_", " "),
            obj.category.replace("_", " "),
        ]
        terms.extend(category_synonyms(obj.category))
        if obj.asset_id:
            terms.append(obj.asset_id.replace("_", " "))
        prompt_map[obj.id] = dedupe_terms(terms)
    return prompt_map


def category_synonyms(category: str) -> list[str]:
    normalized = category.lower().replace("_", " ")
    synonyms = {
        "shelf": ["warehouse shelf", "steel shelf", "storage rack", "metal shelving"],
        "table": ["packing table", "work table", "workbench"],
        "chair": ["office chair", "desk chair", "rolling chair"],
        "box": ["cardboard box", "shipping box", "crate"],
        "cylinder": ["barrel", "drum", "metal barrel", "industrial drum"],
        "cabinet": ["tool chest", "tool cabinet", "rolling tool chest"],
        "tool": ["toolbox", "tool box", "hand tool"],
        "monitor": ["computer monitor", "display screen"],
        "bin": ["storage bin", "plastic bin", "trash can", "metal trash can"],
        "container": ["jerry can", "fuel can", "metal jerry can", "storage container"],
        "bag": ["cement bag", "construction bag", "sack"],
        "hand truck": ["hand truck", "two wheel hand truck", "dolly", "two wheel dolly"],
        "robot arm": ["robot arm", "industrial robot arm"],
        "ladder": ["ladder", "wooden ladder", "metal ladder", "section ladder"],
        "sign": ["safety sign", "wet floor sign", "warning sign"],
        "light": ["fluorescent light", "industrial lamp", "hanging light", "mounted light"],
        "door": ["roller shutter door", "roll up door", "warehouse door", "garage door"],
        "utility box": ["power box", "utility box", "electrical box", "control panel"],
        "duct": ["air duct", "rectangular air duct", "vent duct"],
        "pipe": ["industrial pipes", "metal pipes", "utility pipes"],
        "cable": ["electric cables", "electrical cables", "wall cables"],
        "barrier": ["concrete barrier", "safety barrier", "chainlink fence", "fence panel"],
        "camera": ["security camera", "surveillance camera", "wall camera"],
    }
    return synonyms.get(normalized, [])


def dedupe_terms(terms: list[str]) -> list[str]:
    unique = []
    seen = set()
    for term in terms:
        cleaned = " ".join(term.lower().split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def select_non_overlapping_detection(boxes_xyxy: object, logits: object, selected_boxes: list[np.ndarray]) -> int | None:
    order = np.argsort(-logits.detach().cpu().numpy()).tolist()
    for index in order:
        candidate = boxes_xyxy[index].detach().cpu().numpy().astype(float)
        if all(box_iou(candidate, existing) < 0.65 for existing in selected_boxes):
            return int(index)
    return int(order[0]) if order else None


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def match_phrase_to_object(phrase: str, scene: SceneSpec, assigned: set[str]) -> str | None:
    phrase_lower = phrase.lower()
    candidates = []
    for obj in scene.objects:
        if obj.id in assigned:
            continue
        terms = {obj.category.lower(), obj.id.lower().replace("_", " ")}
        if obj.name:
            terms.add(obj.name.lower())
        overlap = sum(1 for term in terms if term and term in phrase_lower)
        if overlap:
            candidates.append((overlap, obj.id))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def crop_with_mask(image: Image.Image, mask: np.ndarray, box_xyxy: np.ndarray) -> Image.Image:
    x0, y0, x1, y1 = [int(round(value)) for value in box_xyxy.tolist()]
    x0 = max(0, min(x0, image.width - 1))
    y0 = max(0, min(y0, image.height - 1))
    x1 = max(x0 + 1, min(x1, image.width))
    y1 = max(y0 + 1, min(y1, image.height))
    crop = image.crop((x0, y0, x1, y1)).convert("RGBA")
    alpha = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255))
    crop.putalpha(alpha)
    return crop


def draw_segmentation_overlay(image: Image.Image, detections: list[DetectionSpec]) -> Image.Image:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        x0, y0, x1, y1 = detection.box_xyxy
        label = detection.object_id or detection.phrase
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=3)
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 64, 64))
    return overlay
