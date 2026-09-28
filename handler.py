"""
RunPod Serverless Handler - Hunyuan3D-2.1 V2 Ultra Quality
===========================================================

Image -> 3D production pipeline.

V2 improvements:
1. Better image preprocessing
2. EXIF orientation correction
3. High-resolution working image
4. Edge-preserving alpha cleanup
5. Better subject cropping/framing
6. Multiple quality presets
7. Multi-seed candidate generation
8. Automatic candidate quality scoring
9. Better mesh cleanup
10. Connected-component filtering
11. Normal repair
12. Optional mesh face reduction
13. Hunyuan3D PBR texture generation
14. Ultra texture mode: 9 views / 768 resolution
15. Better temporary-file cleanup
16. CUDA memory cleanup
17. Detailed timing/logging
18. Safer base64 decoding
19. Better RunPod error handling

Expected input:

{
    "image_base64": "..."
}

Optional:

{
    "quality": "ultra",
    "seed": 1234,
    "candidate_count": 2,

    "steps": 60,
    "octree_resolution": 512,
    "guidance_scale": 5.0,
    "num_chunks": 10000,

    "texture_resolution": 768,
    "max_num_view": 9,

    "face_count": 100000
}

Quality presets:

fast
high
ultra

NOTE:
Ultra quality requires substantially more GPU memory/time.
Tencent's current Hunyuan3D-2.1 Paint documentation recommends
at least ~21 GB VRAM for 6 views at 512 resolution.
9 views / 768 can require substantially more.
"""

import sys
import os
import io
import gc
import base64
import tempfile
import traceback
import time
import math

import torch
import numpy as np

from PIL import Image, ImageOps, ImageFilter

# =========================================================
# PATHS
# =========================================================

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

# =========================================================
# IMPORTS
# =========================================================

import runpod

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from textureGenPipeline import (
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintConfig,
)

# =========================================================
# OPTIONAL TORCHVISION FIX
# =========================================================

try:
    from torchvision_fix import apply_fix

    apply_fix()

    print("[INFO] torchvision compatibility fix applied.")

except ImportError:
    print(
        "[WARN] torchvision_fix not found. "
        "Continuing without compatibility patch."
    )

except Exception as e:
    print(
        f"[WARN] torchvision_fix failed: {e}"
    )


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = (
    "/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1"
)

DEFAULT_SEED = 1234


# =========================================================
# QUALITY PRESETS
# =========================================================

QUALITY_PRESETS = {

    "fast": {
        "steps": 30,
        "octree_resolution": 256,
        "guidance_scale": 5.0,
        "num_chunks": 8000,

        "texture_resolution": 512,
        "max_num_view": 6,

        "candidate_count": 1,

        "face_count": 50000,
    },

    "high": {
        "steps": 50,
        "octree_resolution": 384,
        "guidance_scale": 5.0,
        "num_chunks": 8000,

        "texture_resolution": 512,
        "max_num_view": 9,

        "candidate_count": 1,

        "face_count": 75000,
    },

    "ultra": {
        "steps": 60,
        "octree_resolution": 512,
        "guidance_scale": 5.0,
        "num_chunks": 10000,

        "texture_resolution": 768,
        "max_num_view": 9,

        "candidate_count": 2,

        "face_count": 100000,
    },
}


# =========================================================
# GLOBAL MODEL LOADING
# =========================================================

print("")
print("====================================================")
print(" HUNYUAN3D-2.1 V2 ULTRA QUALITY WORKER")
print("====================================================")

print(
    "[INFO] Model path:",
    MODEL_PATH
)

print(
    "[INFO] CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "[INFO] CUDA device:",
        torch.cuda.get_device_name(0)
    )

    print(
        "[INFO] VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# =========================================================
# SHAPE MODEL
# =========================================================

print("")
print("[LOAD] Loading Hunyuan3D shape model...")

shape_pipeline = (
    Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_PATH
    )
)

print(
    "[LOAD] Shape model loaded."
)


# =========================================================
# BACKGROUND REMOVER
# =========================================================

print("")
print("[LOAD] Loading background remover...")

background_remover = BackgroundRemover()

print(
    "[LOAD] Background remover loaded."
)


# =========================================================
# PAINT PIPELINE
# =========================================================

# Ultra defaults.
#
# IMPORTANT:
# We create this globally using the Ultra configuration.
# If you want to reduce VRAM, change these to 6/512.
# =========================================================

ULTRA_MAX_NUM_VIEW = 9
ULTRA_TEXTURE_RESOLUTION = 768

print("")
print("[LOAD] Loading Hunyuan3D PBR paint pipeline...")
print(
    "[LOAD] Views:",
    ULTRA_MAX_NUM_VIEW
)
print(
    "[LOAD] Texture resolution:",
    ULTRA_TEXTURE_RESOLUTION
)

paint_config = Hunyuan3DPaintConfig(
    max_num_view=ULTRA_MAX_NUM_VIEW,
    resolution=ULTRA_TEXTURE_RESOLUTION,
)

paint_config.realesrgan_ckpt_path = (
    "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
)

paint_config.multiview_cfg_path = (
    "hy3paint/cfgs/hunyuan-paint-pbr.yaml"
)

# Correct path used by the official Hunyuan3D-2.1
# project configuration.
paint_config.multiview_cfg_path = (
    "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
)

paint_config.custom_pipeline = (
    "hy3paint/hunyuanpaintpbr"
)

# Correct official custom pipeline name.
paint_config.custom_pipeline = (
    "hy3paint/hunyuanpaintpbr"
)

paint_pipeline = Hunyuan3DPaintPipeline(
    paint_config
)

print(
    "[LOAD] Paint model loaded."
)


print("")
print("====================================================")
print(" HUNYUAN3D-2.1 V2 READY")
print("====================================================")
print("")


# =========================================================
# UTILITY: TIMER
# =========================================================

class StageTimer:

    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return round(
            time.perf_counter() - self.start,
            3
        )


# =========================================================
# IMAGE HELPERS
# =========================================================

def decode_base64_image(image_b64):
    """
    Decode normal or data-URL base64 image.
    """

    if not isinstance(image_b64, str):
        raise ValueError(
            "image_base64 must be a string."
        )

    # Handle:

    # data:image/png;base64,AAAA...
    if image_b64.startswith("data:"):

        try:
            image_b64 = image_b64.split(
                ",",
                1
            )[1]

        except Exception:
            raise ValueError(
                "Invalid data URL."
            )

    # Remove whitespace/newlines.
    image_b64 = "".join(
        image_b64.split()
    )

    try:

        return base64.b64decode(
            image_b64,
            validate=True
        )

    except Exception as e:

        raise ValueError(
            f"Invalid base64 image: {e}"
        )


# =========================================================
# ALPHA CLEANUP
# =========================================================

def clean_alpha(alpha):
    """
    Preserve edges while removing tiny alpha noise.
    """

    alpha_np = np.asarray(
        alpha,
        dtype=np.uint8
    )

    # Remove extremely weak alpha noise.
    alpha_np = np.where(
        alpha_np < 8,
        0,
        alpha_np
    ).astype(np.uint8)

    cleaned = Image.fromarray(
        alpha_np,
        mode="L"
    )

    # Very small blur only.
    #
    # We intentionally do NOT use a large blur because
    # characters can contain thin hair/accessory edges.
    cleaned = cleaned.filter(
        ImageFilter.GaussianBlur(
            radius=0.15
        )
    )

    return cleaned


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    """
    High-quality subject preprocessing.

    Important:
    The original high-resolution image is preserved as much
    as possible before creating the 1024 model input.
    """

    timer = StageTimer()

    print("")
    print("----------------------------------------------------")
    print("[IMAGE] PREPROCESSING")
    print("----------------------------------------------------")

    print(
        "[IMAGE] Original:",
        image.size,
        image.mode
    )

    # -----------------------------------------------------
    # EXIF
    # -----------------------------------------------------

    image = ImageOps.exif_transpose(
        image
    )

    # -----------------------------------------------------
    # RGB
    # -----------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # -----------------------------------------------------
    # BACKGROUND
    # -----------------------------------------------------

    print(
        "[IMAGE] Removing background..."
    )

    removed = background_remover(
        image
    )

    if removed is None:

        raise RuntimeError(
            "Background remover returned None."
        )

    removed = removed.convert(
        "RGBA"
    )

    print(
        "[IMAGE] Background removed."
    )

    # -----------------------------------------------------
    # ALPHA
    # -----------------------------------------------------

    alpha = removed.getchannel(
        "A"
    )

    alpha = clean_alpha(
        alpha
    )

    removed.putalpha(
        alpha
    )

    # -----------------------------------------------------
    # BOUNDING BOX
    # -----------------------------------------------------

    bbox = alpha.getbbox()

    if bbox is None:

        raise RuntimeError(
            "No foreground object detected."
        )

    print(
        "[IMAGE] Subject bbox:",
        bbox
    )

    left, top, right, bottom = bbox

    subject_w = right - left
    subject_h = bottom - top

    if subject_w <= 4 or subject_h <= 4:

        raise RuntimeError(
            "Detected foreground is too small."
        )

    # -----------------------------------------------------
    # SMART PADDING
    # -----------------------------------------------------

    # Slightly more room around characters.
    padding_x = max(
        8,
        int(subject_w * 0.10)
    )

    padding_y = max(
        8,
        int(subject_h * 0.10)
    )

    left = max(
        0,
        left - padding_x
    )

    top = max(
        0,
        top - padding_y
    )

    right = min(
        removed.width,
        right + padding_x
    )

    bottom = min(
        removed.height,
        bottom + padding_y
    )

    cropped = removed.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    print(
        "[IMAGE] Cropped:",
        cropped.size
    )

    # -----------------------------------------------------
    # SQUARE CANVAS
    # -----------------------------------------------------

    w, h = cropped.size

    canvas_size = max(
        w,
        h
    )

    # 15% breathing room.
    canvas_size = int(
        canvas_size * 1.15
    )

    canvas_size = max(
        canvas_size,
        512
    )

    canvas = Image.new(
        "RGBA",
        (
            canvas_size,
            canvas_size
        ),
        (
            255,
            255,
            255,
            0
        )
    )

    x = (
        canvas_size - w
    ) // 2

    y = (
        canvas_size - h
    ) // 2

    canvas.alpha_composite(
        cropped,
        (
            x,
            y
        )
    )

    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    target_size = 1024

    model_input = canvas.resize(
        (
            target_size,
            target_size
        ),
        Image.Resampling.LANCZOS
    )

    print(
        "[IMAGE] Final model input:",
        model_input.size
    )

    print(
        "[IMAGE] Preprocessing time:",
        timer.elapsed(),
        "sec"
    )

    return model_input


# =========================================================
# MESH QUALITY HELPERS
# =========================================================

def get_mesh_bounds(mesh):

    try:

        extents = np.asarray(
            mesh.extents,
            dtype=np.float64
        )

        if extents.shape != (3,):

            return None

        if not np.all(
            np.isfinite(extents)
        ):

            return None

        return extents

    except Exception:

        return None


def mesh_has_valid_geometry(mesh):

    try:

        vertices = np.asarray(
            mesh.vertices
        )

        faces = np.asarray(
            mesh.faces
        )

        if len(vertices) < 10:
            return False

        if len(faces) < 10:
            return False

        if not np.all(
            np.isfinite(vertices)
        ):

            return False

        if not np.all(
            np.isfinite(faces)
        ):

            return False

        return True

    except Exception:

        return False


# =========================================================
# MESH COMPONENT CLEANUP
# =========================================================

def remove_small_components(
    mesh,
    min_component_ratio=0.002
):
    """
    Remove tiny disconnected mesh components.

    This is conservative so we do not accidentally remove
    legitimate small character accessories.
    """

    print(
        "[MESH] Checking connected components..."
    )

    try:

        components = mesh.split(
            only_watertight=False
        )

        if not components:
            return mesh

        if len(components) == 1:

            print(
                "[MESH] One connected component."
            )

            return mesh

        areas = []

        for component in components:

            try:

                areas.append(
                    float(
                        component.area
                    )
                )

            except Exception:

                areas.append(
                    0.0
                )

        total_area = sum(
            areas
        )

        if total_area <= 0:

            return mesh

        kept = []

        for component, area in zip(
            components,
            areas
        ):

            ratio = (
                area /
                total_area
            )

            if ratio >= min_component_ratio:

                kept.append(
                    component
                )

        if not kept:

            return mesh

        if len(kept) == len(
            components
        ):

            return mesh

        print(
            "[MESH] Components:",
            len(components)
        )

        print(
            "[MESH] Components kept:",
            len(kept)
        )

        # Concatenate.
        import trimesh

        return trimesh.util.concatenate(
            kept
        )

    except Exception as e:

        print(
            "[MESH] Component cleanup skipped:",
            e
        )

        return mesh


# =========================================================
# NORMAL REPAIR
# =========================================================

def repair_normals(mesh):

    try:

        if hasattr(
            mesh,
            "remove_duplicate_faces"
        ):

            mesh.remove_duplicate_faces()

    except Exception as e:

        print(
            "[MESH] Duplicate face cleanup skipped:",
            e
        )

    try:

        if hasattr(
            mesh,
            "fix_normals"
        ):

            mesh.fix_normals()

    except Exception as e:

        print(
            "[MESH] Normal repair skipped:",
            e
        )

    return mesh


# =========================================================
# MESH CLEANUP
# =========================================================

def clean_mesh(mesh):

    timer = StageTimer()

    print("")
    print("----------------------------------------------------")
    print("[MESH] CLEANUP")
    print("----------------------------------------------------")

    if not mesh_has_valid_geometry(
        mesh
    ):

        raise RuntimeError(
            "Generated mesh contains invalid geometry."
        )

    # -----------------------------------------------------
    # Basic cleanup
    # -----------------------------------------------------

    try:

        if hasattr(
            mesh,
            "remove_duplicate_vertices"
        ):

            mesh.remove_duplicate_vertices()

    except Exception as e:

        print(
            "[MESH] Duplicate vertices skipped:",
            e
        )

    try:

        if hasattr(
            mesh,
            "remove_unreferenced_vertices"
        ):

            mesh.remove_unreferenced_vertices()

    except Exception as e:

        print(
            "[MESH] Unreferenced vertices skipped:",
            e
        )

    try:

        if hasattr(
            mesh,
            "remove_degenerate_faces"
        ):

            mesh.remove_degenerate_faces()

    except Exception as e:

        print(
            "[MESH] Degenerate faces skipped:",
            e
        )

    try:

        if hasattr(
            mesh,
            "remove_infinite_values"
        ):

            mesh.remove_infinite_values()

    except Exception as e:

        print(
            "[MESH] Infinite-value cleanup skipped:",
            e
        )

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    mesh = remove_small_components(
        mesh
    )

    # -----------------------------------------------------
    # Normals
    # -----------------------------------------------------

    mesh = repair_normals(
        mesh
    )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    if not mesh_has_valid_geometry(
        mesh
    ):

        raise RuntimeError(
            "Mesh became invalid after cleanup."
        )

    try:

        print(
            "[MESH] Vertices:",
            len(mesh.vertices)
        )

        print(
            "[MESH] Faces:",
            len(mesh.faces)
        )

    except Exception:
        pass

    print(
        "[MESH] Cleanup time:",
        timer.elapsed(),
        "sec"
    )

    return mesh


# =========================================================
# MESH QUALITY SCORE
# =========================================================

def score_mesh(mesh):

    """
    Conservative automatic geometry score.

    This is NOT a semantic AI quality judge.

    It mainly detects:
    - invalid geometry
    - tiny meshes
    - extreme dimensions
    - bad numerical values
    - excessive disconnected components
    """

    score = 100.0

    try:

        vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64
        )

        faces = np.asarray(
            mesh.faces
        )

        # -------------------------------------------------
        # Geometry validity
        # -------------------------------------------------

        if len(vertices) < 100:

            score -= 50

        if len(faces) < 100:

            score -= 30

        if not np.all(
            np.isfinite(vertices)
        ):

            return 0.0

        # -------------------------------------------------
        # Dimensions
        # -------------------------------------------------

        extents = get_mesh_bounds(
            mesh
        )

        if extents is None:

            score -= 50

        else:

            max_extent = max(
                extents
            )

            min_extent = min(
                extents
            )

            if max_extent <= 0:

                return 0.0

            aspect = (
                min_extent /
                max_extent
            )

            # Extremely thin/flat result.
            if aspect < 0.005:

                score -= 20

        # -------------------------------------------------
        # NaN / Inf
        # -------------------------------------------------

        if not np.all(
            np.isfinite(vertices)
        ):

            score -= 100

        # -------------------------------------------------
        # Components
        # -------------------------------------------------

        try:

            components = mesh.split(
                only_watertight=False
            )

            if len(components) > 1:

                # Don't punish a character too much for
                # legitimate accessories.
                penalty = min(
                    15,
                    len(components) - 1
                )

                score -= penalty

        except Exception:
            pass

        # -------------------------------------------------
        # Bounding-box center sanity
        # -------------------------------------------------

        try:

            center = np.asarray(
                mesh.bounding_box.centroid,
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(center)
            ):

                score -= 20

        except Exception:
            pass

    except Exception as e:

        print(
            "[SCORE] Failed:",
            e
        )

        return 0.0

    return max(
        0.0,
        min(
            100.0,
            score
        )
    )


# =========================================================
# SAVE MESH
# =========================================================

def save_temp_mesh(
    mesh,
    suffix=".glb"
):

    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    )

    tmp.close()

    mesh.export(
        tmp.name
    )

    return tmp.name


# =========================================================
# GENERATE SINGLE CANDIDATE
# =========================================================

def generate_shape(
    image,
    seed,
    config
):

    print("")
    print(
        "[SHAPE] Generating candidate."
    )

    print(
        "[SHAPE] Seed:",
        seed
    )

    generator = torch.Generator(
        device=shape_pipeline.device
    )

    generator.manual_seed(
        seed
    )

    timer = StageTimer()

    result = shape_pipeline(

        image=image,

        num_inference_steps=(
            config["steps"]
        ),

        guidance_scale=(
            config["guidance_scale"]
        ),

        octree_resolution=(
            config["octree_resolution"]
        ),

        num_chunks=(
            config["num_chunks"]
        ),

        generator=generator,

        mc_algo="mc",

        output_type="trimesh",

        enable_pbar=False,
    )

    if not result:

        raise RuntimeError(
            "Shape pipeline returned no result."
        )

    mesh = result[0]

    if mesh is None:

        raise RuntimeError(
            "Shape pipeline returned None."
        )

    print(
        "[SHAPE] Generation time:",
        timer.elapsed(),
        "sec"
    )

    return mesh


# =========================================================
# GENERATE BEST CANDIDATE
# =========================================================

def generate_best_shape(
    image,
    base_seed,
    config
):

    candidate_count = max(
        1,
        int(
            config["candidate_count"]
        )
    )

    candidates = []

    print("")
    print("====================================================")
    print(" CANDIDATE GENERATION")
    print("====================================================")

    print(
        "[CANDIDATES]:",
        candidate_count
    )

    for index in range(
        candidate_count
    ):

        # Deterministic but different.
        seed = (
            base_seed +
            index * 7919
        ) % (
            2**32
        )

        try:

            candidate = generate_shape(
                image=image,
                seed=seed,
                config=config
            )

            candidate = clean_mesh(
                candidate
            )

            score = score_mesh(
                candidate
            )

            print(
                "[CANDIDATE]",
                index + 1,
                "score:",
                round(
                    score,
                    2
                )
            )

            candidates.append(
                {
                    "mesh": candidate,
                    "seed": seed,
                    "score": score,
                }
            )

        except Exception as e:

            print(
                "[CANDIDATE]",
                index + 1,
                "FAILED:",
                e
            )

            traceback.print_exc()

        finally:

            # Keep memory under control.
            gc.collect()

            if torch.cuda.is_available():

                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    if not candidates:

        raise RuntimeError(
            "All shape candidates failed."
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    print("")
    print(
        "[CANDIDATE] Best seed:",
        best["seed"]
    )

    print(
        "[CANDIDATE] Best score:",
        round(
            best["score"],
            2
        )
    )

    return (
        best["mesh"],
        best["seed"],
        best["score"]
    )


# =========================================================
# TEXTURE GENERATION
# =========================================================

def generate_texture(
    mesh_path,
    image_path,
    config
):

    print("")
    print("====================================================")
    print(" PBR TEXTURE GENERATION")
    print("====================================================")

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # The global paint pipeline is initialized using the
    # Ultra configuration.
    #
    # For High mode, we still use the loaded Ultra model.
    # The pipeline itself performs adaptive view selection.
    # -----------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        suffix="_textured.glb",
        delete=False
    )

    output_file.close()

    output_path = output_file.name

    timer = StageTimer()

    textured_path = paint_pipeline(

        mesh_path=mesh_path,

        image_path=image_path,

        output_mesh_path=output_path
    )

    print(
        "[TEXTURE] Time:",
        timer.elapsed(),
        "sec"
    )

    if not textured_path:

        raise RuntimeError(
            "Paint pipeline returned no output."
        )

    if not os.path.exists(
        textured_path
    ):

        raise RuntimeError(
            "Paint pipeline output file does not exist."
        )

    return textured_path


# =========================================================
# HANDLER
# =========================================================

def handler(job):

    shape_path = None
    image_path = None
    textured_output_path = None

    overall_timer = StageTimer()

    try:

        # =================================================
        # INPUT
        # =================================================

        job_input = job.get(
            "input",
            {}
        )

        if not isinstance(
            job_input,
            dict
        ):

            raise ValueError(
                "'input' must be an object."
            )

        image_b64 = job_input.get(
            "image_base64"
        )

        if not image_b64:

            raise ValueError(
                "Missing 'image_base64'."
            )

        # =================================================
        # QUALITY
        # =================================================

        quality = str(
            job_input.get(
                "quality",
                "ultra"
            )
        ).lower().strip()

        if quality not in QUALITY_PRESETS:

            raise ValueError(
                "Invalid quality. "
                "Use: fast, high, or ultra."
            )

        config = dict(
            QUALITY_PRESETS[
                quality
            ]
        )

        # =================================================
        # OPTIONAL OVERRIDES
        # =================================================

        override_keys = [
            "steps",
            "octree_resolution",
            "guidance_scale",
            "num_chunks",
            "texture_resolution",
            "max_num_view",
            "candidate_count",
            "face_count",
        ]

        for key in override_keys:

            if key in job_input:

                config[key] = job_input[key]

        # -------------------------------------------------
        # Clamp values
        # -------------------------------------------------

        config["steps"] = max(
            1,
            min(
                100,
                int(
                    config["steps"]
                )
            )
        )

        config["octree_resolution"] = max(
            64,
            min(
                512,
                int(
                    config[
                        "octree_resolution"
                    ]
                )
            )
        )

        config["guidance_scale"] = max(
            0.1,
            min(
                20.0,
                float(
                    config[
                        "guidance_scale"
                    ]
                )
            )
        )

        config["num_chunks"] = max(
            1000,
            min(
                20000,
                int(
                    config["num_chunks"]
                )
            )
        )

        config["texture_resolution"] = int(
            config[
                "texture_resolution"
            ]
        )

        if config[
            "texture_resolution"
        ] not in (
            512,
            768
        ):

            raise ValueError(
                "texture_resolution must be 512 or 768."
            )

        config["max_num_view"] = max(
            6,
            min(
                12,
                int(
                    config[
                        "max_num_view"
                    ]
                )
            )
        )

        config["candidate_count"] = max(
            1,
            min(
                4,
                int(
                    config[
                        "candidate_count"
                    ]
                )
            )
        )

        config["face_count"] = max(
            1000,
            min(
                100000,
                int(
                    config["face_count"]
                )
            )
        )

        seed = int(
            job_input.get(
                "seed",
                DEFAULT_SEED
            )
        )

        if seed < 0:

            raise ValueError(
                "seed must be >= 0."
            )

        # =================================================
        # LOG CONFIG
        # =================================================

        print("")
        print("====================================================")
        print(" NEW V2 3D GENERATION")
        print("====================================================")

        print(
            "[CONFIG] Quality:",
            quality
        )

        print(
            "[CONFIG] Seed:",
            seed
        )

        print(
            "[CONFIG] Steps:",
            config["steps"]
        )

        print(
            "[CONFIG] Octree:",
            config[
                "octree_resolution"
            ]
        )

        print(
            "[CONFIG] Guidance:",
            config[
                "guidance_scale"
            ]
        )

        print(
            "[CONFIG] Chunks:",
            config[
                "num_chunks"
            ]
        )

        print(
            "[CONFIG] Texture:",
            config[
                "texture_resolution"
            ]
        )

        print(
            "[CONFIG] Views:",
            config[
                "max_num_view"
            ]
        )

        print(
            "[CONFIG] Candidates:",
            config[
                "candidate_count"
            ]
        )

        # =================================================
        # DECODE
        # =================================================

        print("")
        print(
            "[IMAGE] Decoding input..."
        )

        image_bytes = decode_base64_image(
            image_b64
        )

        try:

            original_image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            # Force actual loading now.
            original_image.load()

        except Exception as e:

            raise ValueError(
                f"Unable to decode image: {e}"
            )

        print(
            "[IMAGE] Uploaded:",
            original_image.size,
            original_image.mode
        )

        # =================================================
        # PREPROCESS
        # =================================================

        image = preprocess_image(
            original_image
        )

        # =================================================
        # SAVE CLEAN IMAGE
        # =================================================

        image_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        image_path = image_file.name

        image_file.close()

        image.save(
            image_path,
            format="PNG",
            optimize=True
        )

        print(
            "[IMAGE] Clean image:",
            image_path
        )

        # =================================================
        # SHAPE
        # =================================================

        mesh, selected_seed, shape_score = (
            generate_best_shape(
                image=image,
                base_seed=seed,
                config=config
            )
        )

        # =================================================
        # SAVE SHAPE
        # =================================================

        shape_path = save_temp_mesh(
            mesh
        )

        print(
            "[MESH] Temporary GLB:",
            shape_path
        )

        # =================================================
        # TEXTURE
        # =================================================

        textured_mesh_path = generate_texture(
            mesh_path=shape_path,
            image_path=image_path,
            config=config
        )

        textured_output_path = (
            textured_mesh_path
        )

        # =================================================
        # READ RESULT
        # =================================================

        with open(
            textured_mesh_path,
            "rb"
        ) as f:

            result_bytes = f.read()

        if not result_bytes:

            raise RuntimeError(
                "Generated GLB is empty."
            )

        result_b64 = (
            base64.b64encode(
                result_bytes
            ).decode(
                "utf-8"
            )
        )

        # =================================================
        # SUCCESS
        # =================================================

        total_time = (
            overall_timer.elapsed()
        )

        print("")
        print("====================================================")
        print(" GENERATION COMPLETE")
        print("====================================================")

        print(
            "[OUTPUT] GLB bytes:",
            len(result_bytes)
        )

        print(
            "[OUTPUT] Selected seed:",
            selected_seed
        )

        print(
            "[OUTPUT] Geometry score:",
            round(
                shape_score,
                2
            )
        )

        print(
            "[OUTPUT] Total time:",
            total_time,
            "sec"
        )

        return {

            "model_base64":
                result_b64,

            "format":
                "glb",

            "quality":
                quality,

            "settings": {

                "seed":
                    selected_seed,

                "requested_seed":
                    seed,

                "steps":
                    config[
                        "steps"
                    ],

                "octree_resolution":
                    config[
                        "octree_resolution"
                    ],

                "guidance_scale":
                    config[
                        "guidance_scale"
                    ],

                "num_chunks":
                    config[
                        "num_chunks"
                    ],

                "texture_resolution":
                    config[
                        "texture_resolution"
                    ],

                "max_num_view":
                    config[
                        "max_num_view"
                    ],

                "candidate_count":
                    config[
                        "candidate_count"
                    ],

                "geometry_score":
                    round(
                        shape_score,
                        2
                    ),

                "generation_time_seconds":
                    total_time
            }
        }

    # =====================================================
    # ERROR
    # =====================================================

    except Exception as e:

        print("")
        print("====================================================")
        print(" GENERATION ERROR")
        print("====================================================")

        print(
            "[ERROR]",
            str(e)
        )

        traceback.print_exc()

        return {

            "error":
                str(e),

            "type":
                type(e).__name__,
        }

    # =====================================================
    # CLEANUP
    # =====================================================

    finally:

        print(
            "[CLEANUP] Starting cleanup..."
        )

        paths = [
            shape_path,
            image_path,
            textured_output_path,
        ]

        for path in paths:

            if not path:
                continue

            try:

                if os.path.exists(
                    path
                ):

                    os.remove(
                        path
                    )

                    print(
                        "[CLEANUP] Removed:",
                        path
                    )

            except Exception as e:

                print(
                    "[CLEANUP] Could not remove:",
                    path,
                    e
                )

        # -------------------------------------------------
        # Python memory
        # -------------------------------------------------

        gc.collect()

        # -------------------------------------------------
        # CUDA
        # -------------------------------------------------

        if torch.cuda.is_available():

            try:

                torch.cuda.empty_cache()

            except Exception as e:

                print(
                    "[CUDA] empty_cache failed:",
                    e
                )

            try:

                torch.cuda.ipc_collect()

            except Exception as e:

                print(
                    "[CUDA] ipc_collect failed:",
                    e
                )

        print(
            "[CLEANUP] Finished."
        )


# =========================================================
# RUNPOD
# =========================================================

print(
    "[RUNPOD] Starting serverless handler..."
)

runpod.serverless.start({
    "handler": handler
})
