# Scenethesis MVP Technical Progress Report

Date: 2026-06-16

## Current State

The repo now has a strict paper-faithful path in addition to the original lightweight MVP path. The strict path is the one relevant to recreating Scenethesis.

Current faithful command target:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py --prompt "a busy warehouse storage and packing area with two steel storage shelves, many cardboard boxes, wooden crates, plastic crates, barrels, a hand truck, a metal trash can, a tool chest, a packing table, tools, a ladder, jerry cans, and a chair" --out runs/warehouse_faithful_rich_007 --repair-rounds 2
```

Latest successful output directory:

```text
runs/warehouse_faithful_rich_007
```

Latest validation summary:

- 19 objects
- 0 broad collision count from final metrics
- 0 floating objects
- 0 unsupported objects
- SDF optimizer status: `ok`
- Judge `needs_repair`: `false`
- Final artifacts include `scene_spec.json`, `scene.glb`, `scene.usd`, `render.png`, `metrics.json`, `judge.json`, `repair_history.json`, and `report.md`
- The judge gave prompt alignment `0.75` and noted imperfect visibility of some jerry cans/tools even though those objects exist in `scene_spec.json`.

## Implemented Method

The faithful runner now follows this local sequence:

1. Runtime validation checks OpenAI, Blender, CUDA, required Python packages, required checkpoints, CLIP index, asset registry, and disk availability.
2. OpenAI planner returns a strict Pydantic `SceneSpec`.
3. Planner validation enforces category counts and subtype requirements from the prompt, including cardboard boxes, wooden crates, plastic crates, and trash cans.
4. OpenAI image generation creates `guidance.png`.
5. GroundingDINO and SAM segment the guidance image into per-object detections, masks, and crops.
6. Depth Pro estimates depth and writes `depth.npy`, `depth_preview.png`, and metadata.
7. The point-cloud stage projects masks through depth and writes `scene_graph_3d.json` plus object point clouds.
8. CLIP retrieval uses image crop embeddings, object text, and registry metadata overlap to choose mesh-backed assets.
9. Layout relation normalization converts open-shelf children from `inside` to `on`, because open racks are support surfaces rather than enclosed containers.
10. SDF/PyTorch3D optimization processes the scene hierarchy and writes `sdf_optimizer.json`.
11. Blender imports real local glTF/glb assets, frames the full warehouse scene, renders `render.png`, and exports `scene.glb` plus `scene.usd` where available.
12. OpenAI vision judge evaluates the rendered scene and returns strict JSON scores/actions.
13. Repair actions are validated before use. Invalid, impossible, or no-op actions fail the run instead of being locally converted into success.
14. Final outputs are written only after planning, guidance, segmentation, depth, retrieval, optimization, rendering, judging, and optional repair complete.

## Removed Substitute Success Behavior

Removed or hardened:

- Local planner substitute.
- Local judge substitute.
- Pillow preview renderer substitute.
- Manual GLB substitute exporter.
- CLI/config substitute flags.
- Local judge reconciliation that dismissed judge failures after the fact.
- No-op repair acceptance when `needs_repair=true`.
- Resume mode that could proceed with stale/missing artifacts.
- PyBullet support-contact substitute that reported a non-simulation check as a PyBullet result.

Current behavior is fail-fast and writes `failure.json` when the strict pipeline cannot complete.

## Asset Retrieval Work

Implemented asset work:

- Curated local Poly Haven warehouse pack.
- Source/license metadata in `assets/manifests/polyhaven_warehouse_assets.json`.
- Registry entries in `configs/warehouse_asset_registry.yaml`.
- Thumbnail rendering and CLIP index at `assets/indexes/warehouse_clip_index.npz`.
- Retrieval scoring now combines:
  - CLIP image score from segmentation crop.
  - CLIP text score from planned object id/name/description/category.
  - Deterministic registry metadata score from asset name/tags.

Recent fix:

- Generic metadata tokens such as `bin` and `can` no longer help retrieval. A requested trash can must carry the `trash` trait and should select the metal trash-can asset over a generic plastic storage container.

Remaining gap:

- The warehouse pack is still too small compared with a paper-quality Objaverse subset. The pipeline can run, but visual diversity and composition richness are constrained by available meshes.

## Physics And Optimization Work

Implemented:

- Runtime requirement for PyTorch3D and CUDA in faithful mode.
- SDF optimizer artifacts and progress reporting.
- Broad-phase AABB skip before expensive signed-distance queries.
- Registry-declared support planes for tables/shelves where mesh SDF support contact is not reliable for open structures.
- Support validation for shelf children using support heights.
- Sibling spreading across support planes to reduce shelf/table clutter.
- Deterministic SDF free-slot search for floor/support objects that remain trapped after local signed-distance updates.
- Strict metrics conversion from `sdf_optimizer.json`.

Failure behavior:

- If SDF optimization reports a non-`ok` status, final metrics report a collision count rather than hiding the failure.
- Missing real meshes fail scene runtime validation.
- Missing CUDA or PyTorch3D fails runtime validation.
- In the latest run, `jerrycan_01` initially failed SDF collision resolution; the added deterministic search checked 12 mesh-query candidates and accepted the first zero-penetration slot.

Remaining gap:

- RoMa correspondence optimization is not yet deeply wired into the main optimization loop. The next faithful step should use rendered object views plus guidance crops to optimize scale/yaw/translation from 2D and 3D correspondence losses before or during SDF optimization.

## Judge Work

Implemented:

- OpenAI vision judge receives prompt, scene spec, render, metrics, and semantic constraints.
- Judge responses must be strict JSON.
- Judge response schema now constrains the OpenAI output to known score keys and a flat action schema accepted by OpenAI structured outputs.
- Repair actions are validated against scene object IDs and allowed relations.
- `set_parent` only accepts support relations `on` or `inside`.
- `move_near` must target a valid object.
- No-op actions are rejected when `needs_repair=true`.
- The judge prompt now states that non-actionable residual issues should lower scores/notes, not create no-op repair actions.
- The judge-facing action set no longer advertises `set_parent`; support changes should use `change_relation` with `relation=on|inside`, avoiding invalid `set_parent(..., relation=near)` responses.

Observed failure:

- `gpt-4.1` repeatedly returned no-op repair actions for a scene whose relationships were already satisfied. The strict validator correctly failed those runs.

Current config:

- `OPENAI_VISION_MODEL` can override config.
- `configs/scenethesis_faithful.yaml` sets the vision judge model to `gpt-4o`, which succeeded on the latest run.

## Verified Runs

Useful output:

- `runs/warehouse_faithful_rich_007`

Key artifacts in that run:

- `coarse_scene_spec.json`
- `guidance.png`
- `segmentation.json`
- `depth.json`
- `scene_graph_3d.json`
- `clip_retrieval.json`
- `sdf_optimizer.json`
- `scene_spec.json`
- `scene.glb`
- `scene.usd`
- `render.png`
- `metrics.json`
- `judge.json`
- `report.md`

Known visual issue:

- The current render is much richer than the earlier shelf/barrel/table screenshot, but it remains below the paper examples because the local asset database, occlusion-aware composition, and RoMa pose-alignment stage are still limited. This is an asset/data and pose-alignment problem more than a missing high-level pipeline stage at this point.

## Test Coverage

Tests cover:

- Pydantic schemas.
- Asset registry loading.
- Mesh path resolution.
- Prompt category and subtype requirements.
- Trash-can trait preservation.
- Metadata token filtering for retrieval.
- Collision math.
- Support and shelf relation behavior.
- Layout optimizer behavior.
- Strict faithful runtime/resume validation.
- Judge validation, including invalid actions, no-op rejection, and mixed actionable/no-op repair cleanup.
- Mounted warehouse support semantics for ceiling and wall assets.

Tests intentionally do not call OpenAI.

## Remaining Work For Paper Fidelity

Highest-impact next steps:

1. Expand the asset database with warehouse-specific meshes: pallets, forklifts, conveyors, pallet racks, industrial bins, scanners, robot arms, safety barriers, carts, wrapped pallet loads, and varied boxes.
2. Add an Objaverse/HF-style manifest and controlled downloader with license metadata, not a bulk download.
3. Integrate RoMa correspondence optimization into the main pose refinement loop.
4. Render object-level alignment views for each object and store correspondence diagnostics.
5. Add multi-view final judging against both guidance image and rendered scene views.
6. Improve scene composition: wall props, floor markings, lighting fixtures, dense but non-colliding shelf contents, and visible aisle structure.
7. Add failure qualification artifacts so a run can be marked `unqualified` without being counted as accepted.

## Load Estimate

For a 12-18 object warehouse scene on 8 GB VRAM:

- GroundingDINO: 3-5 GB VRAM.
- SAM ViT-H: 5-8 GB VRAM.
- Depth Pro: 3-6 GB VRAM.
- CLIP retrieval: 2-4 GB VRAM.
- RoMa: 5-8 GB VRAM when fully integrated.
- SDF/PyTorch3D: 4-8 GB VRAM depending on sample count and object count.
- Blender render/export: 2-6 GB VRAM if GPU rendering is used.

The practical runtime model is sequential subprocess stages, batch size 1, cached artifacts, and no simultaneous residency of all heavy models.

## 2026-06-16 Addendum: Asset, RoMa, Multi-View, Qualification Pass

Implemented since the previous report:

- Added `configs/hf_simready_warehouse_manifest.yaml` as a controlled HF/SimReady asset manifest. Bulk download is explicitly disabled.
- Added `scripts/import_hf_simready_assets.py`, which downloads only declared asset directories and declared USD payload dependencies, converts USD to GLB via Blender, and writes `assets/manifests/hf_simready_warehouse_assets.json`.
- Imported nine additional licensed warehouse assets from `nvidia/PhysicalAI-SimReady-Warehouse-01`: forklift, pallet, wrapped pallet load, pallet rack, platform cart, barcode scanner, plastic crate, safety barrier, and safety tape.
- Rebuilt `configs/warehouse_asset_registry.yaml`; it now has 74 assets, including 9 HF SimReady entries.
- Rebuilt `assets/indexes/warehouse_clip_index.npz` against the expanded registry.
- Added explicit RoMa/DINOv2 checkpoint paths and updated `scripts/download_faithful_checkpoints.py` to download them.
- Added RoMa rendered-object correspondence diagnostics in `src/scenethesis_mvp/optimization/roma_correspondence.py`.
- Added Blender-generated alternate scene views and per-object alignment views.
- Added multi-image OpenAI judging using primary render, guidance image, and alternate scene views.
- Added `qualification.json` so completed runs can be marked `accepted` or `unqualified`.

Observed failures and fixes:

- The first HF import failed on a wrapped pallet assembly because the USD referenced payloads outside the primary directory. Fixed by declaring dependency prefixes in the manifest.
- The first rack import missed the rack skeleton and one cardboard-box dependency. Fixed by declaring `RackLargeEmpty_A1` and `Cardbox_D1`.
- The initial cart assembly had too many nested payload dependencies for the controlled subset. Replaced it with a direct platform-cart USD from the same licensed dataset.

Verification:

- Runtime check passes with RoMa weights, DINOv2 weights, CUDA, Blender, PyTorch3D, GroundingDINO, SAM, DepthPro, OpenCLIP, and expanded asset registry.
- Tests pass: 38 passed.
- Renderer check on `runs/warehouse_faithful_rich_007` produced `render.png`, four scene views, and 19 per-object alignment views.
- RoMa check on that existing scene produced `correspondence_diagnostics.json` with `ok=true`, `failed_object_count=0`, and 19 bounded yaw updates.

Additional implementation and run status:

- Fixed a real semantic failure where pallet-scale floor objects, including pallets, wrapped pallet loads, forklifts, carts, barriers, and floor markings, could be treated as supported children. They are now normalized to floor-scale `near` relations instead of invalid `on`/`inside` support relations.
- Fixed mounted-object support handling across SDF optimization, layout stability, and mesh validation. Assets tagged `ceiling` or `wall` are now validated as mounted objects instead of being incorrectly judged as floating floor objects.
- Added OpenAI retry handling for rate-limit windows and switched multi-image vision calls to low-detail image payloads to reduce token pressure while preserving the real judge stage.
- Added deterministic judge repair sanitation: no-op repair actions are dropped only when at least one actionable repair remains. All-no-op repair outputs still fail validation instead of being counted as success.
- Fresh HF-guided run `runs/warehouse_hf_simready_001` completed end-to-end through retrieval, SDF, Blender, RoMa, multi-view judge, and qualification.
- `runs/warehouse_hf_simready_001` produced 20 objects, `collision_count=0`, `floating_count=0`, `unsupported_count=0`, `support_penalty=0.0`, `render_visual_support_failure_count=0`, and RoMa diagnostics with `ok=true`, `failed_object_count=0`, `applied_updates=18`.
- The same run is correctly marked `unqualified`, not accepted, because the OpenAI judge returned `needs_repair=true`. Judge scores were: object category accuracy 0.6, orientation alignment 0.7, physical plausibility 0.8, prompt alignment 0.5, spatial coherence 0.5.
- The final render is asset-rich but still visually below the paper examples. The main visible issues are overbright/washed-out imported geometry, large-object occlusion, weak aisle composition, and judge-requested relation repairs for mounted lights/camera.

## 2026-06-16 Addendum: Paper-Style Warehouse Presentation Pass

Implemented after comparing the current render against Figure 1-style Scenethesis examples:

- Added `src/scenethesis_mvp/layout/warehouse_staging.py`, a deterministic warehouse presentation layout seed applied after asset retrieval and after repair edits, before SDF optimization.
- The staging pass keeps the real selected assets, but moves them into a coherent room composition: rack against the back wall, forklift/cart/pallets in the foreground/midground, table to the side, wall and ceiling fixtures mounted, and shelf/table children placed on support surfaces.
- Added scale normalization for oversized forklift/rack assets so one imported mesh does not dominate or crop the whole frame.
- Fixed support snapping so wall/ceiling fixtures stay mounted, while ground-and-wall doors still snap to the floor.
- Updated SDF support-sibling placement to bias shelf children toward the visible front of rack shelves.
- Reworked Blender presentation rendering:
  - primary render now uses a lower, wider, front-facing presentation camera instead of a high oblique inventory camera;
  - imported warehouse assets get semantic materials when imported USD/GLB materials are white, missing, or visually unusable;
  - rack legs/frames render as green metal, shelves/pallets as wood, boxes as cardboard, carts as aluminum/rubber, forklift as orange, and safety objects as yellow/orange;
  - default wall is warm neutral instead of the previous green factory-wall texture;
  - lighting is warmer and less overexposed, with stronger ambient occlusion.

Verification artifacts:

- `runs/warehouse_visual_stage_003/render.png` shows the staged render without running another OpenAI judge.
- `runs/warehouse_visual_sdf_001/render.png` shows the same staged scene after the real SDF optimizer and Blender render.
- `runs/warehouse_visual_sdf_001/sdf_optimizer.json`: `status=ok`, 20 objects, no failed objects.
- `runs/warehouse_visual_sdf_001/render_validation.json`: `ok=true`, `visual_support_failure_count=0`.
- Tests pass: 39 passed.

Remaining gap:

- The scene is now much closer in composition to the paper examples, but it is still not paper-grade photorealistic output. Remaining issues are asset-library limitations, limited decorative/background object density, simplified material shading, and no full multi-view judge acceptance pass after this visual staging change.

## 2026-06-16 Addendum: Dense Accepted Warehouse Run And Renderer Debug Pass

Implemented in this pass:

- Fixed imported material handling in `src/scenethesis_mvp/render/blender_script.py`.
  - Poly Haven / textured GLTF materials are now preserved instead of being blindly cleared.
  - HF SimReady USD-to-GLB assets are still assigned semantic materials because many imported source materials are flat white or shared across unrelated mesh parts.
  - The renderer now adds procedural color variation, bump, roughness, bevels, and weighted normals to reduce the flat/plastic look.
  - Floor texture maps from `configs/scenethesis_faithful.yaml` are used; the green factory-wall texture is disabled unless explicitly requested because it produced worse warehouse renders than the neutral procedural wall.
- Tightened the presentation camera so the primary render focuses on floor-level warehouse composition instead of over-framing ceiling/wall utilities.
- Expanded dense warehouse planning requirements before image guidance/segmentation:
  - pallet rack/shelf, forklift, platform cart, hand truck, ladder, packing table, scanner, tools, tool chest/cabinet, pallets, wrapped pallet load, barrels, jerry cans, bin, safety tape, barrier, lights, door, pipes, utility box, camera, duct, safety sign, and six shelf/floor boxes/crates.
- Added concrete `required_instance_plan` payloads for the OpenAI planner so dense runs ask for explicit instance ids such as `cardboard_box_01`, `wooden_crate_01`, `jerrycan_01`, `barcode_scanner_01`, and `tool_01`.
- Fixed repair-loop/staging interaction:
  - Staging now preserves spatial constraint targets instead of dropping them during constraint rebuild.
  - If the judge asks to move objects near the packing table, staging now honors those target constraints after the generic warehouse layout pass.
- Added/used `pipeline_diagnostics.json` as the top-level correctness summary for anchor, asset assignment, segmentation, scene graph, SDF/collision/support losses, visual support, RoMa, and judge state.

Important planner finding:

- `gpt-4o-mini` repeatedly failed the dense strict planner contract by omitting required small table objects (`scanner`, `tool`) or under-counting dense boxes/containers.
- `gpt-4o` passed the same strict contract for the dense warehouse prompt with 34 objects. For dense faithful runs, the practical recommendation is to set `OPENAI_MODEL=gpt-4o` for planning. This is not a fallback; it is a stricter model choice needed to satisfy the enforced SceneSpec contract.

Accepted run:

- Run directory: `runs/warehouse_dense_full_002`
- Prompt: dense warehouse aisle with pallet racks, boxes/crates, forklift, pallets, wrapped pallet load, barrels, jerry cans, bins, cart, hand truck, ladder, packing table, scanner/tools, tool chest, safety tape/barriers, industrial lights, wall utilities, camera, and roller shutter door.
- Object count: 34
- Category counts:
  - shelf 1, box 6, forklift 1, pallet 2, pallet_load 1, cylinder/barrels 2, container/jerry cans 2, cart 1, hand_truck 1, ladder 1, table 1, scanner 1, tool 2, cabinet/tool chest 1, floor_marking 1, barrier 1, light 2, door 1, pipe 1, utility_box 1, camera 1, duct 1, bin 1, sign 1.

Final metrics:

- `collision_count=0`
- `floating_count=0`
- `unsupported_count=0`
- `boundary_violations=0`
- `collision_penalty=0.0`
- `support_penalty=0.0`
- `relation_penalty=0.0`
- `total_penalty=0.0`

Pipeline diagnostics:

- Anchor check: `pallet_rack_01`, ok.
- Asset assignment: no missing asset ids.
- Segmentation coverage: 34 detections, no missing objects.
- Scene graph coverage: 34 point clouds, no missing objects.
- SDF optimizer: status ok, no failed objects.
- Render support validation: `visual_support_failure_count=0`.
- RoMa correspondence: ok, `failed_object_count=0`, `applied_updates=30`.
- OpenAI judge: `needs_repair=false`.

Qualification:

- `qualification.json`: `status=accepted`, `accepted=true`.
- Required outputs exist: `scene_spec.json`, `scene.glb`, `scene.usd`, `render.png`, `metrics.json`, `judge.json`, `pipeline_diagnostics.json`, `qualification.json`, and `report.md`.
- `python -m pytest`: 41 passed.
- Output validator passes with `PYTHONPATH=src python -m scenethesis_mvp.pipeline.validate_outputs runs/warehouse_dense_full_002`.

Known residual limitations:

- The accepted render is substantially denser and more coherent, but still below the paper examples in photorealism.
- Remaining limitations are mostly asset/camera/render quality: limited asset variety, some partially cropped right-side props, simplified material/shadow model, and no generative texture synthesis.
- The scene.glb is large for a laptop workflow, about 105 MB; scene.usd is about 278 MB.
- Windows RoMa runs without the local-correlation custom kernel, so correspondence works but is slower than an equivalent Linux/CUDA setup.

## 2026-06-17 Addendum: Mesh-Derived Shelf Support Fix

Reason for this pass:

- The accepted dense warehouse run still had visually floating shelf objects.
- Root cause was not a renderer-only issue. The SDF optimizer and Blender visual support validator were both accepting rack children against registry `support_heights`.
- The imported HF pallet rack mesh does not actually expose three shelf surfaces matching the registry metadata. It has two broad rack-board support planes in the rendered geometry, so boxes placed on the third/metadata level looked unsupported.

Implemented:

- Fixed GLB/GLTF physics mesh loading in `src/scenethesis_mvp/optimization/sdf_optimizer.py`.
  - Trimesh was consuming raw glTF axes while Blender imports glTF as Z-up.
  - The SDF loader now converts GLB/GLTF vertices into Blender-compatible axes before normalization, so collision/support checks use the same geometry that is rendered.
- Added mesh-derived support plane extraction in the SDF optimizer.
  - `MeshTemplate` now stores local support planes.
  - `PlacedMesh` now stores world-space support planes.
  - Container/shelf parents use broad, slab-like mesh components as support surfaces.
  - Parent-child support now reports `mesh_derived_support_plane` when geometry is available.
  - If a container mesh has no derived support plane, the optimizer raises an error instead of silently accepting metadata.
- Updated sibling staging inside SDF optimization to seed child placement from mesh-derived support planes when available.
- Updated Blender render validation in `src/scenethesis_mvp/render/blender_script.py`.
  - Visual support validation now derives support planes from the rendered object group itself.
  - Children on imported mesh containers validate against `mesh_derived_parent_support_plane`.
  - If a rendered container parent has no usable mesh support plane, validation fails with `missing_mesh_support_plane`.
- Added tests covering:
  - imported rack support plane extraction from mesh geometry, not registry height count;
  - mesh-derived support target preference over registry support metadata.

Verification:

- Test suite: `43 passed`.
- New diagnostic run: `runs/warehouse_mesh_support_001`.
- SDF optimizer:
  - `sdf_optimizer.json`: `status=ok`, 34 objects.
  - 9 supported children used mesh-derived support planes.
- Blender visual support:
  - `render_validation.json`: `ok=true`, `visual_support_failure_count=0`.
  - Shelf child support examples:
    - `cardboard_box_01`: bottom `0.397319`, target `0.39711`, error `0.000209`.
    - `cardboard_box_02`: bottom `1.800714`, target `1.800505`, error `0.000209`.
    - `wooden_crate_01`: bottom `0.397121`, target `0.39711`, error `0.000011`.
    - `plastic_crate_02`: bottom `1.800505`, target `1.800505`, error `0.0`.
- Render output: `runs/warehouse_mesh_support_001/render.png`.

Run status:

- The geometry, SDF, RoMa refinement, Blender render, and render support validation stages completed.
- The run is intentionally **not accepted** because the final OpenAI vision judge failed with HTTP 401.
- The provided `.env.example` value is the placeholder `sk-your-****here`, not a valid OpenAI API key.
- `qualification.json` correctly marks the run `unqualified` at stage `judge`.
- `metrics.json`, `scene_spec.json`, and final `report.md` were not produced for this run because the final acceptance path did not complete.

Remaining issues:

- The shelf support bug shown by the user is fixed by geometry-backed validation, not hidden by looser thresholds.
- Composition still needs work: foreground objects are dense, the forklift is partially cropped, and some right-side props are clipped.
- The rack asset itself has two usable shelf boards after Blender import, not the three shelf levels implied by registry metadata. Better warehouse renders need either a different rack mesh with more real support levels or a cleaned rack asset without baked-in small boxes.
- A valid OpenAI API key is required to complete the final judge and produce an accepted run.

## 2026-06-17 Addendum: Depth-Pose And RoMa Artifact Integration

Implemented in this pass:

- Added `src/scenethesis_mvp/vision/depth_pose_refinement.py`.
  - Uses `scene_graph_3d.json` point-cloud bounding boxes from Grounded-SAM masks plus Depth Pro.
  - Applies bounded metric scale updates only when height and footprint scale estimates agree.
  - Applies bounded yaw updates only when the depth box is directionally meaningful.
  - Snaps objects back to valid support after pose changes.
  - Writes `depth_pose_refinement.json` with per-object before/update/after records.
- Wired depth-pose refinement into `scripts/run_faithful.py` through `run_faithful_pipeline`.
  - The stage runs after asset retrieval and warehouse presentation staging, before SDF/PyTorch3D.
  - Repair rounds rerun the same depth-pose stage before SDF.
- Tightened qualification and diagnostics.
  - `pipeline_diagnostics.json` now includes a `depth_pose_refinement` check.
  - `qualification.json` now requires `depth_pose_refinement.json` for accepted faithful runs.
  - `report.md` now separates Depth Pose Refinement from RoMa Correspondence.
- Improved RoMa diagnostics.
  - RoMa now writes `correspondences/<object_id>.npz` with guidance/rendered keypoints and confidence.
  - RoMa writes per-object JSON summaries under `correspondences/`.
  - RoMa writes `pose_alignment_history.json` with before/update/after placement records.
  - Scale is intentionally not inferred from RoMa object alignment views because those views are orthographically recentered and scale-normalized per object. Metric scale now comes from Depth Pro point-cloud boxes instead.

Smoke verification:

- Runtime gate passes with CUDA, PyTorch3D, Grounded-SAM, SAM, Depth Pro, RoMa, CLIP index, Blender, and checkpoints.
- Depth-pose smoke on `runs/warehouse_gpt55_full_001` produced `runs/depth_pose_smoke_003/depth_pose_refinement.json`.
- The conservative default applied 4 scale updates and 23 yaw updates on that 33-object scene.
- Full test suite passes: `46 passed`.

Remaining gap:

- The pose loop is closer to the paper than before because metric depth now affects scale/yaw before SDF and RoMa writes real correspondence artifacts. It is still not the full Scenethesis joint optimization objective: the next step is to combine rendered scene projection, RoMa correspondences, and Depth Pro point clouds into a single iterative 5-DoF loss for translation, yaw, and scale instead of running bounded refinement stages around SDF.
