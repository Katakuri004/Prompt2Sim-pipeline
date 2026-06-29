from __future__ import annotations

from collections import Counter
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
    min_expected_box_iou: float = 0.10
    min_expected_box_coverage: float = 0.20


class GroundedSAMSegmenter:
    def __init__(self, config: GroundedSAMConfig):
        self.config = config

    def segment(
        self,
        image_path: str | Path,
        scene: SceneSpec,
        out_dir: str | Path,
        expected_boxes_norm: dict[str, list[float]] | None = None,
    ) -> SegmentationResult:
        self._validate_files()
        expected_boxes_norm = expected_boxes_norm or {}
        expected_ids = {obj.id for obj in scene.objects}
        if set(expected_boxes_norm) != expected_ids:
            raise RuntimeError(
                "Grounded-SAM requires exact GPT guidance-box coverage: "
                f"missing={sorted(expected_ids - set(expected_boxes_norm))}, "
                f"extra={sorted(set(expected_boxes_norm) - expected_ids)}"
            )
        target_dir = Path(out_dir)
        masks_dir = target_dir / "masks"
        crops_dir = target_dir / "crops"
        dino_crops_dir = target_dir / "dino_crops"
        masks_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        dino_crops_dir.mkdir(parents=True, exist_ok=True)

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
        sam_prompt_boxes_xyxy = []
        selected_dino_boxes_xyxy = []
        selected_logits = []
        selected_phrases = []
        selected_object_ids = []
        selected_guidance_ious = []
        selected_guidance_coverages = []
        selected_box_sources = []
        evidence_records: list[dict[str, object]] = []
        category_counts = Counter(obj.category.strip().lower() for obj in scene.objects)
        support_parent_ids = {obj.parent_id for obj in scene.objects if obj.parent_id}

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
                evidence_records.append(
                    {
                        "object_id": obj.id,
                        "prompts": prompts,
                        "status": "no_groundingdino_candidates",
                    }
                )
                continue
            boxes_xyxy = box_convert(
                boxes=boxes * torch.tensor([image_width, image_height, image_width, image_height], device=boxes.device),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            expected_norm = np.asarray(expected_boxes_norm[obj.id], dtype=float)
            expected_xyxy = expected_norm * np.asarray([image_width, image_height, image_width, image_height], dtype=float)
            selection = select_semantic_detection(
                boxes_xyxy,
                logits,
                expected_xyxy=expected_xyxy,
                min_expected_box_iou=self.config.min_expected_box_iou,
                min_expected_box_coverage=self.config.min_expected_box_coverage,
            )
            proposal_origin = "full_image"
            crop_diagnostics: dict[str, object] = {}
            if selection is None and category_counts[obj.category.strip().lower()] > 1:
                crop_bounds = expanded_pixel_crop(expected_xyxy, image_width, image_height)
                crop_path = dino_crops_dir / f"{obj.id}.png"
                Image.fromarray(image_source).crop(crop_bounds).save(crop_path)
                _, crop_tensor = load_image(str(crop_path))
                crop_height = crop_bounds[3] - crop_bounds[1]
                crop_width = crop_bounds[2] - crop_bounds[0]
                crop_boxes, crop_logits, crop_phrases = [], [], []
                for box_threshold, text_threshold in [
                    (self.config.box_threshold, self.config.text_threshold),
                    (self.config.retry_box_threshold, self.config.retry_text_threshold),
                ]:
                    crop_boxes, crop_logits, crop_phrases = predict(
                        model=model,
                        image=crop_tensor,
                        caption=caption,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.config.device,
                    )
                    if len(crop_boxes) > 0:
                        break
                crop_diagnostics = {
                    "crop_path": str(crop_path),
                    "crop_bounds_xyxy": list(crop_bounds),
                    "crop_proposal_count": len(crop_boxes),
                }
                if len(crop_boxes) > 0:
                    crop_boxes_xyxy = box_convert(
                        boxes=crop_boxes
                        * torch.tensor(
                            [crop_width, crop_height, crop_width, crop_height],
                            device=crop_boxes.device,
                        ),
                        in_fmt="cxcywh",
                        out_fmt="xyxy",
                    )
                    crop_boxes_xyxy = crop_boxes_xyxy + crop_boxes_xyxy.new_tensor(
                        [crop_bounds[0], crop_bounds[1], crop_bounds[0], crop_bounds[1]]
                    )
                    crop_selection = select_semantic_detection(
                        crop_boxes_xyxy,
                        crop_logits,
                        expected_xyxy=expected_xyxy,
                        min_expected_box_iou=self.config.min_expected_box_iou,
                        min_expected_box_coverage=self.config.min_expected_box_coverage,
                    )
                    crop_diagnostics["crop_candidates"] = detection_candidate_records(
                        crop_boxes_xyxy,
                        crop_logits,
                        crop_phrases,
                        expected_xyxy,
                    )
                    if crop_selection is not None:
                        boxes_xyxy = crop_boxes_xyxy
                        logits = crop_logits
                        phrases = crop_phrases
                        selection = crop_selection
                        proposal_origin = "expected_guidance_box_crop"
            if selection is None:
                evidence_records.append(
                    {
                        "object_id": obj.id,
                        "prompts": prompts,
                        "proposal_count": len(boxes),
                        "status": "semantic_overlap_below_threshold",
                        "required_iou": self.config.min_expected_box_iou,
                        "required_smaller_box_coverage": self.config.min_expected_box_coverage,
                        "candidates": detection_candidate_records(
                            boxes_xyxy,
                            logits,
                            phrases,
                            expected_xyxy,
                        ),
                        **crop_diagnostics,
                    }
                )
                continue
            selected_index, guidance_iou, guidance_coverage = selection
            dino_box_xyxy = boxes_xyxy[selected_index]
            prompt_box_array, box_source = select_sam_prompt_box(
                expected_xyxy=expected_xyxy,
                dino_xyxy=dino_box_xyxy.detach().cpu().numpy().astype(float),
                category_instance_count=category_counts[obj.category.strip().lower()],
                supports_children=obj.id in support_parent_ids,
            )
            sam_prompt_box = dino_box_xyxy.new_tensor(prompt_box_array)
            sam_prompt_boxes_xyxy.append(sam_prompt_box)
            selected_dino_boxes_xyxy.append(dino_box_xyxy)
            selected_logits.append(logits[selected_index])
            selected_phrases.append(str(phrases[selected_index]))
            selected_object_ids.append(obj.id)
            selected_guidance_ious.append(guidance_iou)
            selected_guidance_coverages.append(guidance_coverage)
            selected_box_sources.append(box_source)
            evidence_records.append(
                {
                    "object_id": obj.id,
                    "prompts": prompts,
                    "proposal_count": len(logits),
                    "proposal_origin": proposal_origin,
                    "status": "semantic_verified",
                    "guidance_box_iou": round(float(guidance_iou), 6),
                    "guidance_box_coverage": round(float(guidance_coverage), 6),
                    "box_source": box_source,
                    "sam_prompt_box_xyxy": [round(float(value), 3) for value in prompt_box_array.tolist()],
                    "dino_box_xyxy": [
                        round(float(value), 3)
                        for value in dino_box_xyxy.detach().cpu().numpy().astype(float).tolist()
                    ],
                    **crop_diagnostics,
                }
            )

        if not sam_prompt_boxes_xyxy:
            all_prompts = [" / ".join(prompts) for prompts in object_prompt_map.values()]
            result = SegmentationResult(
                image_path=str(image_path),
                image_width=image_width,
                image_height=image_height,
                detections=[],
                missing_object_ids=[obj.id for obj in scene.objects],
            )
            write_json(target_dir / "segmentation.json", result)
            write_segmentation_evidence(target_dir, evidence_records)
            raise RuntimeError("GroundingDINO found no objects for planned object prompts: " + "; ".join(all_prompts))

        sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(self.config.sam_checkpoint))
        sam.to(device=self.config.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image_source)
        boxes_xyxy_tensor = torch.stack(sam_prompt_boxes_xyxy).to(self.config.device)
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
            box = sam_prompt_boxes_xyxy[index].detach().cpu().numpy().astype(float)
            mask_np = clip_mask_to_box(mask_np, box)
            mask_area = int(mask_np.sum())
            if mask_area < self.config.min_mask_pixels:
                update_evidence_status(
                    evidence_records,
                    object_id,
                    "sam_mask_too_small",
                    mask_area=mask_area,
                    required_mask_pixels=self.config.min_mask_pixels,
                )
                continue
            dino_box = selected_dino_boxes_xyxy[index].detach().cpu().numpy().astype(float)
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
                    guidance_box_iou=round(float(selected_guidance_ious[index]), 6),
                    guidance_box_coverage=round(float(selected_guidance_coverages[index]), 6),
                    box_source=selected_box_sources[index],
                    box_xyxy=[round(float(value), 3) for value in box.tolist()],
                    dino_box_xyxy=[round(float(value), 3) for value in dino_box.tolist()],
                    mask_path=str(mask_path),
                    crop_path=str(crop_path),
                    mask_area=mask_area,
                )
            )
            update_evidence_status(evidence_records, object_id, "mask_accepted", mask_area=mask_area)

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
        write_segmentation_evidence(target_dir, evidence_records)
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


def expanded_pixel_crop(
    expected_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
    padding_fraction: float = 0.45,
    min_padding_pixels: int = 24,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in expected_xyxy.tolist()]
    padding_x = max(float(min_padding_pixels), (x1 - x0) * padding_fraction)
    padding_y = max(float(min_padding_pixels), (y1 - y0) * padding_fraction)
    left = max(0, int(np.floor(x0 - padding_x)))
    top = max(0, int(np.floor(y0 - padding_y)))
    right = min(image_width, int(np.ceil(x1 + padding_x)))
    bottom = min(image_height, int(np.ceil(y1 + padding_y)))
    if right <= left or bottom <= top:
        raise RuntimeError(f"Invalid expected guidance crop: {expected_xyxy.tolist()}")
    return left, top, right, bottom


def detection_candidate_records(
    boxes_xyxy: object,
    logits: object,
    phrases: list[str],
    expected_xyxy: np.ndarray,
) -> list[dict[str, object]]:
    candidate_boxes = boxes_xyxy.detach().cpu().numpy().astype(float)
    candidate_scores = logits.detach().cpu().numpy().astype(float)
    return [
        {
            "box_xyxy": [round(float(value), 3) for value in box.tolist()],
            "score": round(float(candidate_scores[index]), 6),
            "phrase": str(phrases[index]),
            "guidance_iou": round(float(box_iou(box, expected_xyxy)), 6),
            "guidance_smaller_box_coverage": round(float(box_smaller_coverage(box, expected_xyxy)), 6),
        }
        for index, box in enumerate(candidate_boxes)
    ]


def select_semantic_detection(
    boxes_xyxy: object,
    logits: object,
    expected_xyxy: np.ndarray,
    min_expected_box_iou: float,
    min_expected_box_coverage: float = 0.20,
) -> tuple[int, float, float] | None:
    boxes = boxes_xyxy.detach().cpu().numpy().astype(float)
    scores = logits.detach().cpu().numpy().astype(float)
    expected_ious = np.asarray([box_iou(candidate, expected_xyxy) for candidate in boxes], dtype=float)
    expected_coverages = np.asarray(
        [box_smaller_coverage(candidate, expected_xyxy) for candidate in boxes],
        dtype=float,
    )
    order = sorted(
        range(len(boxes)),
        key=lambda index: (expected_ious[index], expected_coverages[index], scores[index]),
        reverse=True,
    )
    for index in order:
        if (
            expected_ious[index] < min_expected_box_iou
            and expected_coverages[index] < min_expected_box_coverage
        ):
            continue
        return int(index), float(expected_ious[index]), float(expected_coverages[index])
    return None


def select_sam_prompt_box(
    expected_xyxy: np.ndarray,
    dino_xyxy: np.ndarray,
    category_instance_count: int,
    supports_children: bool = False,
) -> tuple[np.ndarray, str]:
    if category_instance_count < 1:
        raise ValueError("category_instance_count must be at least 1")
    if category_instance_count > 1:
        return expected_xyxy.copy(), "gpt_instance_bbox_groundingdino_verified"
    if supports_children:
        return expected_xyxy.copy(), "gpt_support_bbox_groundingdino_verified"
    return dino_xyxy.copy(), "groundingdino_box_gpt_verified"


def update_evidence_status(
    records: list[dict[str, object]],
    object_id: str,
    status: str,
    **details: object,
) -> None:
    for record in records:
        if record.get("object_id") == object_id:
            record["status"] = status
            record.update(details)
            return
    raise RuntimeError(f"Missing segmentation evidence record for {object_id}")


def write_segmentation_evidence(target_dir: Path, records: list[dict[str, object]]) -> None:
    write_json(
        target_dir / "segmentation_evidence.json",
        {
            "provider": "gpt_bbox_groundingdino_sam_fusion",
            "records": records,
        },
    )


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


def box_smaller_coverage(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    smaller = min(area_a, area_b)
    return 0.0 if smaller <= 0 else intersection / smaller


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


def clip_mask_to_box(mask: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x0, y0, x1, y1 = [int(round(value)) for value in box_xyxy.tolist()]
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(x0, min(x1, width))
    y1 = max(y0, min(y1, height))
    bounded = np.zeros_like(mask, dtype=bool)
    bounded[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return bounded


def draw_segmentation_overlay(image: Image.Image, detections: list[DetectionSpec]) -> Image.Image:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        x0, y0, x1, y1 = detection.box_xyxy
        label = detection.object_id or detection.phrase
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=3)
        dx0, dy0, dx1, dy1 = detection.dino_box_xyxy
        draw.rectangle((dx0, dy0, dx1, dy1), outline=(64, 160, 255), width=2)
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 64, 64))
    return overlay
