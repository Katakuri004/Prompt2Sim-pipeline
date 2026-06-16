# Paper-Faithful Scenethesis Recreation Plan

Date: 2026-06-16

This document defines the next implementation track for a faithful local recreation of Scenethesis. It intentionally separates paper components from the current MVP approximations.

## Non-Negotiable Direction

The paper-faithful mode must not use replacement components that make the pipeline look complete.

Disable in paper-faithful mode:

- VLM-only scene graph extraction as a replacement for Grounded-SAM plus Depth Pro.
- AABB collision optimization as a replacement for SDF physical optimization.
- Sampled point-distance mesh checks as a replacement for SDF losses.
- Procedural assets as replacements for retrieved mesh assets.
- Poly Haven warehouse pack as a replacement for an Objaverse-style asset database, except as a temporary asset database while clearly labeled.

Keep as valid paper components:

- API-based LLM planning and judge calls.
- OpenAI image generation for image guidance, since the paper uses GPT-4o and image generation in the vision module.
- Blender render/export.

## Paper Pipeline To Implement

Based on Scenethesis Algorithm 1 and method sections:

1. LLM coarse scene planning returns object list, anchor object, hierarchy, and upsampled prompt.
2. Image guidance generation from the upsampled prompt.
3. Grounded-SAM segmentation:
   - Input: guidance image plus object list.
   - Output: boxes, masks, cropped object images.
4. Depth Pro metric depth:
   - Input: guidance image.
   - Output: metric depth map and focal length estimate.
5. Pose extraction:
   - Project segmentation masks through Depth Pro into 3D point clouds.
   - Estimate object 3D bounding boxes.
   - Initialize 5-DoF pose: scale, yaw/upright rotation, translation.
6. Scene graph construction:
   - VLM receives guidance image, object list, masks/crops, depth-derived poses.
   - Output: ground root, anchor, parent/child graph, relations, initial poses.
7. CLIP asset retrieval:
   - Use CLIP ViT-L/14 LAION-style image and text features.
   - Query local asset database embeddings.
   - Fail if no mesh asset is available for a required object.
8. Physics-aware optimization:
   - BFS over scene graph: anchor, parents, children.
   - Render object/scene RGB and depth.
   - Use RoMa dense correspondences between guidance crop/image and rendered view.
   - Optimize 5-DoF pose with 2D correspondence loss and 3D point-cloud loss.
   - Build and query SDFs for collision and stability losses.
   - Use SGD, not Adam.
9. Multi-view judge:
   - Render multiple views.
   - VLM evaluates object category accuracy, orientation alignment, and spatial coherence.
   - If not qualified, regenerate/replan rather than silently accepting the scene.

Paper parameter targets:

- RoMa correspondences: `m = 100`, confidence threshold `tau >= 0.6`.
- Surface samples for physical state: `n = 400` points per object per optimization iteration.
- 5-DoF variables: scale `s`, upright rotation/yaw `R`, translation `T = (tx, ty, tz)`.
- Joint loss: pose, translation collision, scale collision, and stability.

## Hardware Reality For Current Laptop

Target machine assumption:

- GPU: 8 GB VRAM
- System RAM: 16 GB
- Disk budget for this track: expect 15 to 30 GB, excluding Blender and generated runs
- CUDA required for practical runtime

The laptop can run this if models are loaded sequentially and aggressively unloaded between stages. It should not keep Grounded-SAM, Depth Pro, RoMa, CLIP, Blender, and SDF tensors resident at the same time.

Recommended runtime shape:

- Separate subprocess per heavy model stage.
- Batch size 1.
- FP16 or AMP where model code supports it.
- 1024 px guidance images for the first faithful warehouse target.
- Cache every artifact to disk: masks, crops, depth, point clouds, graph, pose states, SDF diagnostics.
- Hard fail on missing model, missing checkpoint, missing CUDA, missing asset, or empty mask.

## Load Estimate Per Scene

For a warehouse scene with 8 to 15 objects:

| Stage | Expected VRAM | System RAM | Disk/checkpoint load | Runtime estimate |
| --- | ---: | ---: | ---: | ---: |
| GroundingDINO text detection | 3 to 5 GB | 2 to 4 GB | 0.7 to 1 GB | 2 to 8 sec GPU |
| SAM ViT-H mask generation | 5 to 8 GB | 3 to 6 GB | about 2.4 GB | 5 to 20 sec GPU |
| Depth Pro | 3 to 6 GB | 3 to 6 GB | 1 to 3 GB | 1 to 6 sec GPU |
| Mask/depth point projection | <1 GB | 1 to 3 GB | artifact only | 5 to 30 sec CPU |
| VLM graph build | API | <1 GB | artifact only | 10 to 60 sec API latency |
| CLIP ViT-L/14 retrieval | 2 to 4 GB | 2 to 4 GB | 1 to 2 GB | <10 sec per scene after indexing |
| RoMa pose alignment | 5 to 8 GB | 3 to 6 GB | 1 to 2 GB | 1 to 4 min total |
| SDF/PyTorch3D optimization | 4 to 8 GB | 4 to 8 GB | code/cache only | 2 to 10 min total |
| Blender render/export | 2 to 6 GB GPU if used | 2 to 6 GB | scene artifacts | 30 sec to 3 min |

Expected end-to-end runtime on 8 GB VRAM:

- Best case, CUDA working, 8 to 10 objects: 8 to 20 minutes.
- Heavier scene, 12 to 15 objects: 20 to 45 minutes.
- CPU-only for vision/depth/correspondence: not practical for iteration, likely 1 to 3 hours or worse.

## Implementation Plan

### Phase 1: Faithful Runtime Gate

Create `configs/scenethesis_faithful.yaml` with:

- `paper_faithful: true`
- `segmentation.provider: grounded_sam`
- `depth.provider: depth_pro`
- `correspondence.provider: roma`
- `physics.provider: sdf_pytorch3d`
- `asset_retrieval.provider: clip_local_index`
- `allow_substitutes: false`

Add startup validation:

- CUDA visible and PyTorch CUDA works.
- Required checkpoints exist.
- Required Python packages import.
- Blender exists.
- Asset database has mesh entries for all selected categories.
- If any check fails, stop before generating final artifacts.

### Phase 2: Grounded-SAM Stage

Add modules:

- `src/scenethesis_mvp/vision/grounded_sam.py`
- `src/scenethesis_mvp/schemas/segmentation.py`

Artifacts:

- `detections.json`
- `masks/*.png`
- `crops/*.png`
- `segmentation_overlay.png`

Acceptance:

- Every required object either has a mask or the run fails with an object-level error.
- No VLM-only mask substitute.

### Phase 3: Depth Pro Stage

Add modules:

- `src/scenethesis_mvp/vision/depth_pro_runner.py`
- `src/scenethesis_mvp/schemas/depth.py`

Artifacts:

- `depth.npy`
- `depth_preview.png`
- `camera_intrinsics.json`

Acceptance:

- Depth is metric and aligned to the guidance image resolution.
- Missing or invalid depth fails the run.

### Phase 4: 3D Scene Graph And Pose Extraction

Add modules:

- `src/scenethesis_mvp/vision/pointcloud.py`
- `src/scenethesis_mvp/vision/pose_extraction.py`
- `src/scenethesis_mvp/schemas/scene_graph_3d.py`

Artifacts:

- `object_pointclouds/*.ply`
- `initial_3dbb.json`
- `scene_graph_3d.json`
- `pose_init_depth.json`

Acceptance:

- 5-DoF initial pose comes from masks plus metric depth, not from VLM 2D hints alone.

### Phase 5: CLIP Asset Retrieval

Add modules:

- `src/scenethesis_mvp/assets/clip_index.py`
- `src/scenethesis_mvp/assets/objaverse_manifest.py`

Artifacts:

- `asset_index.faiss` or `asset_index.npz`
- `retrieval_scores.json`

Acceptance:

- Required asset retrieval must return a local mesh.
- Missing asset fails. It does not create a procedural stand-in.

### Phase 6: RoMa Pose Alignment

Add modules:

- `src/scenethesis_mvp/optimization/roma_alignment.py`
- `src/scenethesis_mvp/render/object_render.py`

Artifacts:

- `correspondences/*.npz`
- `pose_alignment_history.json`
- `render_alignment_views/*.png`

Acceptance:

- Uses RoMa correspondences.
- Uses 2D and 3D correspondence losses.
- Optimizes scale, yaw/upright rotation, and translation.

### Phase 7: SDF/PyTorch3D Physical Optimization

Add modules:

- `src/scenethesis_mvp/physics/sdf.py`
- `src/scenethesis_mvp/optimization/sdf_optimizer.py`

Artifacts:

- `sdf_metrics.json`
- `physics_optimization_history.json`
- `collision_points.json`
- `stability_points.json`

Acceptance:

- Uses SDF queries against object surface samples.
- Uses translation collision loss, scale collision loss, and stability loss.
- Uses SGD.
- Uses `n = 400` surface samples per object target.

### Phase 8: Multi-View Judge And Regeneration

Update judge:

- Judge receives multi-view renders and guidance image.
- Judge scores category accuracy, orientation alignment, and spatial coherence.
- If below threshold, pipeline marks the run unqualified and triggers regeneration/replanning.

Artifacts:

- `judge_multiview.json`
- `qualification.json`

## Main Risks

1. PyTorch3D on native Windows is likely the biggest engineering blocker. A WSL2 Ubuntu CUDA environment is the cleanest path.
2. 8 GB VRAM is tight but workable only with sequential subprocess execution.
3. Exact paper asset database is not public in a directly reproducible form. We need a curated Objaverse-style local subset and a manifest.
4. SAM ViT-H plus GroundingDINO loaded together may exceed VRAM. The faithful implementation should run the same components sequentially, not replace them.
5. RoMa and SDF optimization will dominate runtime.

## Required User Approval Before Implementation

To proceed with the paper-faithful track, approve:

- Installing a CUDA PyTorch environment, preferably WSL2 Ubuntu or a dedicated conda environment.
- Downloading model checkpoints for GroundingDINO, SAM, Depth Pro, RoMa, CLIP, and PyTorch3D dependencies.
- Increasing disk budget to at least 15 GB for checkpoints and cached artifacts.
- Building or downloading a small curated Objaverse-style warehouse asset subset with licenses and manifest.

