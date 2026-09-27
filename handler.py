"""
RunPod Serverless Handler - Hunyuan3D-2.1
=========================================

Improved image -> 3D pipeline.

Main improvements:
1. Proper background removal
2. Subject cropping
3. Transparent-background handling
4. Better subject framing
5. Higher-quality shape generation settings
6. Deterministic seed
7. Mesh cleanup before texturing
8. Hunyuan3D PBR texture generation
9. Better temporary-file cleanup
10. GPU memory cleanup
11. Detailed logging
12. Safer error handling

Expected input:

{
    "image_base64": "...."
}

Optional:

{
    "image_base64": "....",
    "seed": 1234,
    "steps": 50,
    "octree_resolution": 384,
    "guidance_scale": 5.0,
    "num_chunks": 8000,
    "texture_resolution": 512,
    "max_num_view": 6
}
"""

import sys
import os
import io
import gc
import base64
import tempfile
import traceback

import torch
import numpy as np

from PIL import Image, ImageOps, ImageFilter

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import runpod

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from textureGenPipeline import (
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintConfig
)

# ---------------------------------------------------------
# OPTIONAL TORCHVISION FIX
# ---------------------------------------------------------

try:
    from torchvision_fix import apply_fix

    apply_fix()
    print("[INFO] torchvision compatibility fix applied.")

except Exception as e:
    print(f"[WARN] torchvision_fix skipped: {e}")


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_PATH = "/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1"

DEFAULT_SEED = 1234

# Hunyuan3D-2.1 supports higher quality with more inference steps.
DEFAULT_STEPS = 50

# 384 gives a good balance between quality and VRAM.
DEFAULT_OCTREE_RESOLUTION = 384

# Official pipeline uses 5.0 as a common guidance value.
DEFAULT_GUIDANCE_SCALE = 5.0

DEFAULT_NUM_CHUNKS = 8000

DEFAULT_TEXTURE_RESOLUTION = 512

# Tencent's 2.1 worker uses 6 views by default.
DEFAULT_MAX_NUM_VIEW = 6


# ---------------------------------------------------------
# LOAD SHAPE MODEL
# ---------------------------------------------------------

print("")
print("==============================================")
print(" Loading Hunyuan3D-2.1")
print("==============================================")

print(f"[INFO] Model path: {MODEL_PATH}")

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    MODEL_PATH
)

print("[INFO] Shape model loaded.")


# ---------------------------------------------------------
# LOAD BACKGROUND REMOVER
# ---------------------------------------------------------

print("[INFO] Loading background remover...")

background_remover = BackgroundRemover()

print("[INFO] Background remover loaded.")


# ---------------------------------------------------------
# LOAD TEXTURE / PAINT PIPELINE
# ---------------------------------------------------------

print("[INFO] Loading Hunyuan3D paint pipeline...")

paint_config = Hunyuan3DPaintConfig(
    max_num_view=DEFAULT_MAX_NUM_VIEW,
    resolution=DEFAULT_TEXTURE_RESOLUTION
)

paint_config.realesrgan_ckpt_path = (
    "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
)

paint_config.multiview_cfg_path = (
    "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
)

paint_config.custom_pipeline = (
    "hy3dpaint/hunyuanpaintpbr"
)

paint_pipeline = Hunyuan3DPaintPipeline(
    paint_config
)

print("[INFO] Paint pipeline loaded.")

print("")
print("==============================================")
print(" Hunyuan3D-2.1 READY")
print("==============================================")
print("")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    """
    Convert the user's image into a clean object image.

    Steps:
    - RGB conversion
    - EXIF orientation correction
    - background removal
    - alpha cleanup
    - crop around object
    - padding
    - resize
    - white/neutral background outside subject
    """

    print("[IMAGE] Original size:", image.size)
    print("[IMAGE] Original mode:", image.mode)

    # -----------------------------------------------------
    # Fix camera orientation
    # -----------------------------------------------------

    image = ImageOps.exif_transpose(image)

    # -----------------------------------------------------
    # Convert to RGB first
    # -----------------------------------------------------

    image = image.convert("RGB")

    # -----------------------------------------------------
    # Background removal
    #
    # IMPORTANT:
    # Do NOT check image.mode == RGB here.
    #
    # We explicitly call the remover.
    # -----------------------------------------------------

    print("[IMAGE] Removing background...")

    removed = background_remover(image)

    if removed is None:
        raise RuntimeError(
            "Background remover returned None."
        )

    # Make sure we have RGBA.
    removed = removed.convert("RGBA")

    print("[IMAGE] Background removed.")

    # -----------------------------------------------------
    # Get alpha channel
    # -----------------------------------------------------

    alpha = removed.getchannel("A")

    # Slightly smooth the alpha edge.
    alpha = alpha.filter(
        ImageFilter.GaussianBlur(radius=0.3)
    )

    removed.putalpha(alpha)

    # -----------------------------------------------------
    # Find subject bounding box
    # -----------------------------------------------------

    bbox = alpha.getbbox()

    if bbox is None:
        raise RuntimeError(
            "No foreground object detected in image."
        )

    print("[IMAGE] Subject bounding box:", bbox)

    # -----------------------------------------------------
    # Crop tightly around subject
    # -----------------------------------------------------

    left, top, right, bottom = bbox

    width = right - left
    height = bottom - top

    # Add generous padding.
    padding_x = int(width * 0.12)
    padding_y = int(height * 0.12)

    left = max(0, left - padding_x)
    top = max(0, top - padding_y)
    right = min(removed.width, right + padding_x)
    bottom = min(removed.height, bottom + padding_y)

    cropped = removed.crop(
        (left, top, right, bottom)
    )

    print("[IMAGE] Cropped size:", cropped.size)

    # -----------------------------------------------------
    # Put object into a square canvas.
    #
    # This helps the shape model concentrate on the object
    # instead of a huge empty background.
    # -----------------------------------------------------

    w, h = cropped.size

    canvas_size = max(w, h)

    # Additional breathing room.
    canvas_size = int(canvas_size * 1.15)

    canvas = Image.new(
        "RGBA",
        (canvas_size, canvas_size),
        (255, 255, 255, 0)
    )

    x = (canvas_size - w) // 2
    y = (canvas_size - h) // 2

    canvas.alpha_composite(
        cropped,
        (x, y)
    )

    # -----------------------------------------------------
    # Resize to a reasonable input size.
    #
    # Do not make the input unnecessarily huge.
    # -----------------------------------------------------

    target_size = 1024

    canvas = canvas.resize(
        (target_size, target_size),
        Image.Resampling.LANCZOS
    )

    print(
        "[IMAGE] Final model input:",
        canvas.size
    )

    return canvas


# =========================================================
# MESH CLEANUP
# =========================================================

def clean_mesh(mesh):
    """
    Basic geometry cleanup.

    This does NOT try to magically fix bad geometry.
    It removes common floating/degenerate problems.
    """

    print("[MESH] Cleaning mesh...")

    try:

        # Remove duplicated vertices if available.
        if hasattr(mesh, "remove_duplicate_vertices"):
            mesh.remove_duplicate_vertices()

    except Exception as e:
        print(
            "[MESH] duplicate vertex cleanup skipped:",
            e
        )

    try:

        # Remove unreferenced vertices.
        if hasattr(mesh, "remove_unreferenced_vertices"):
            mesh.remove_unreferenced_vertices()

    except Exception as e:
        print(
            "[MESH] unreferenced vertex cleanup skipped:",
            e
        )

    try:

        # Remove degenerate faces.
        if hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()

    except Exception as e:
        print(
            "[MESH] degenerate cleanup skipped:",
            e
        )

    try:

        # Remove infinite / NaN geometry.
        if hasattr(mesh, "remove_infinite_values"):
            mesh.remove_infinite_values()

    except Exception as e:
        print(
            "[MESH] infinite-value cleanup skipped:",
            e
        )

    print("[MESH] Cleanup finished.")

    return mesh


# =========================================================
# SAVE TEMP FILE
# =========================================================

def save_temp_mesh(mesh):
    """
    Save mesh to a temporary GLB.
    """

    tmp = tempfile.NamedTemporaryFile(
        suffix=".glb",
        delete=False
    )

    tmp.close()

    mesh.export(tmp.name)

    return tmp.name


# =========================================================
# HANDLER
# =========================================================

def handler(job):

    shape_path = None
    image_path = None
    textured_output_path = None

    try:

        # -------------------------------------------------
        # READ INPUT
        # -------------------------------------------------

        job_input = job.get("input", {})

        image_b64 = job_input.get(
            "image_base64"
        )

        if not image_b64:
            raise ValueError(
                "Missing 'image_base64' in job input."
            )

        # -------------------------------------------------
        # OPTIONS
        # -------------------------------------------------

        seed = int(
            job_input.get(
                "seed",
                DEFAULT_SEED
            )
        )

        steps = int(
            job_input.get(
                "steps",
                DEFAULT_STEPS
            )
        )

        octree_resolution = int(
            job_input.get(
                "octree_resolution",
                DEFAULT_OCTREE_RESOLUTION
            )
        )

        guidance_scale = float(
            job_input.get(
                "guidance_scale",
                DEFAULT_GUIDANCE_SCALE
            )
        )

        num_chunks = int(
            job_input.get(
                "num_chunks",
                DEFAULT_NUM_CHUNKS
            )
        )

        print("")
        print("==============================================")
        print(" NEW 3D GENERATION")
        print("==============================================")

        print("[CONFIG] Seed:", seed)
        print("[CONFIG] Steps:", steps)
        print(
            "[CONFIG] Octree:",
            octree_resolution
        )
        print(
            "[CONFIG] Guidance:",
            guidance_scale
        )
        print(
            "[CONFIG] Chunks:",
            num_chunks
        )

        # -------------------------------------------------
        # DECODE IMAGE
        # -------------------------------------------------

        try:

            image_bytes = base64.b64decode(
                image_b64
            )

        except Exception as e:

            raise ValueError(
                f"Invalid base64 image: {e}"
            )

        original_image = Image.open(
            io.BytesIO(image_bytes)
        )

        print(
            "[IMAGE] Uploaded:",
            original_image.size,
            original_image.mode
        )

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        image = preprocess_image(
            original_image
        )

        # -------------------------------------------------
        # SAVE PROCESSED IMAGE
        #
        # IMPORTANT:
        # This SAME cleaned image is used for:
        #
        # 1. Shape generation
        # 2. Texture generation
        #
        # -------------------------------------------------

        image_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        image_path = image_file.name

        image_file.close()

        image.save(
            image_path,
            format="PNG"
        )

        print(
            "[IMAGE] Clean image saved:",
            image_path
        )

        # -------------------------------------------------
        # RANDOM GENERATOR
        # -------------------------------------------------

        generator = torch.Generator(
            device=shape_pipeline.device
        )

        generator.manual_seed(seed)

        # -------------------------------------------------
        # GENERATE 3D SHAPE
        # -------------------------------------------------

        print("")
        print("==============================================")
        print(" GENERATING 3D SHAPE")
        print("==============================================")

        mesh = shape_pipeline(
            image=image,

            # More steps generally provide more
            # refinement than very low-step generation.
            num_inference_steps=steps,

            # Hunyuan3D-2.1 guidance.
            guidance_scale=guidance_scale,

            # Higher surface resolution.
            octree_resolution=octree_resolution,

            # Memory/performance control.
            num_chunks=num_chunks,

            # Deterministic result.
            generator=generator,

            # Explicit marching cubes algorithm.
            mc_algo="mc",

            # We want an actual trimesh.
            output_type="trimesh",

            # Keep logs clean.
            enable_pbar=False
        )[0]

        if mesh is None:
            raise RuntimeError(
                "Hunyuan3D returned no mesh."
            )

        print(
            "[MESH] Shape generation complete."
        )

        # -------------------------------------------------
        # CLEAN SHAPE
        # -------------------------------------------------

        mesh = clean_mesh(mesh)

        # -------------------------------------------------
        # SAVE SHAPE
        # -------------------------------------------------

        shape_path = save_temp_mesh(
            mesh
        )

        print(
            "[MESH] Shape saved:",
            shape_path
        )

        # -------------------------------------------------
        # TEXTURE GENERATION
        # -------------------------------------------------

        print("")
        print("==============================================")
        print(" GENERATING PBR TEXTURE")
        print("==============================================")

        textured_output_path = (
            tempfile.NamedTemporaryFile(
                suffix="_textured.glb",
                delete=False
            ).name
        )

        textured_mesh_path = paint_pipeline(
            mesh_path=shape_path,
            image_path=image_path,
            output_mesh_path=textured_output_path
        )

        if not textured_mesh_path:
            raise RuntimeError(
                "Texture pipeline returned no output."
            )

        print(
            "[TEXTURE] Texture generation complete."
        )

        # -------------------------------------------------
        # READ FINAL GLB
        # -------------------------------------------------

        with open(
            textured_mesh_path,
            "rb"
        ) as f:

            result_bytes = f.read()

        if not result_bytes:
            raise RuntimeError(
                "Generated GLB is empty."
            )

        result_b64 = base64.b64encode(
            result_bytes
        ).decode("utf-8")

        print(
            "[OUTPUT] GLB size:",
            len(result_bytes),
            "bytes"
        )

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        print("")
        print("==============================================")
        print(" GENERATION COMPLETE")
        print("==============================================")
        print("")

        return {
            "model_base64": result_b64,
            "format": "glb",

            # Useful for debugging/frontend.
            "settings": {
                "seed": seed,
                "steps": steps,
                "octree_resolution":
                    octree_resolution,
                "guidance_scale":
                    guidance_scale,
                "num_chunks":
                    num_chunks
            }
        }

    except Exception as e:

        print("")
        print("==============================================")
        print(" GENERATION ERROR")
        print("==============================================")

        print(str(e))

        traceback.print_exc()

        return {
            "error": str(e)
        }

    finally:

        # -------------------------------------------------
        # CLEAN TEMP FILES
        # -------------------------------------------------

        for path in [
            shape_path,
            image_path,
            textured_output_path
        ]:

            if path and os.path.exists(path):

                try:
                    os.remove(path)

                except Exception as e:
                    print(
                        "[CLEANUP] Could not remove:",
                        path,
                        e
                    )

        # -------------------------------------------------
        # CLEAN PYTHON MEMORY
        # -------------------------------------------------

        gc.collect()

        # -------------------------------------------------
        # CLEAN CUDA MEMORY
        # -------------------------------------------------

        try:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except Exception as e:

            print(
                "[CUDA] Cleanup skipped:",
                e
            )


# =========================================================
# START RUNPOD SERVERLESS
# =========================================================

print("[RUNPOD] Starting serverless handler...")

runpod.serverless.start({
    "handler": handler
})
