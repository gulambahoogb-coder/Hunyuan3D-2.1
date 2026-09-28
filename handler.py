"""
RunPod Serverless Handler
Hunyuan3D-2.1 - Production High Quality

FINAL PRODUCTION BASELINE

Pipeline:
    Image
      ↓
    EXIF correction
      ↓
    Background removal
      ↓
    Edge-preserving alpha cleanup
      ↓
    Smart character framing
      ↓
    Hunyuan3D-2.1 Shape
      ↓
    Mesh validation
      ↓
    Official Hunyuan3D-2.1 PBR Paint
      ↓
    Textured GLB

Quality-first configuration:
    Shape:
        60 diffusion steps
        octree 512
        guidance 5.0
        chunks 10000

    Texture:
        9 views
        768 resolution

Input:

{
    "image_base64": "..."
}

Optional:

{
    "seed": 1234,

    "steps": 60,
    "octree_resolution": 512,
    "guidance_scale": 5.0,
    "num_chunks": 10000,

    "enable_flashvdm": false
}

Output:

{
    "model_base64": "...",
    "format": "glb"
}
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import sys
import io
import gc
import time
import base64
import tempfile
import traceback

import numpy as np
import torch

from PIL import Image, ImageOps

# =========================================================
# LOCAL HUNYUAN PATHS
# =========================================================

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

# =========================================================
# RUNPOD
# =========================================================

import runpod

# =========================================================
# HUNYUAN IMPORTS
# =========================================================

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import (
    Hunyuan3DDiTFlowMatchingPipeline,
)

from hy3dpaint.textureGenPipeline import (
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintConfig,
)

# =========================================================
# OPTIONAL TORCHVISION COMPATIBILITY
# =========================================================

try:
    from torchvision_fix import apply_fix

    apply_fix()

    print("[INFO] torchvision compatibility fix applied.")

except ImportError:
    print("[INFO] torchvision_fix not installed.")

except Exception as e:
    print(
        "[WARN] torchvision_fix failed:",
        str(e)
    )


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = os.environ.get(
    "HUNYUAN_MODEL_PATH",
    "/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1"
)

DEFAULT_SEED = int(
    os.environ.get(
        "HUNYUAN_DEFAULT_SEED",
        "1234"
    )
)

# ---------------------------------------------------------
# QUALITY-FIRST DEFAULTS
# ---------------------------------------------------------

DEFAULT_STEPS = 60

DEFAULT_OCTREE_RESOLUTION = 512

DEFAULT_GUIDANCE_SCALE = 5.0

DEFAULT_NUM_CHUNKS = 10000

# ---------------------------------------------------------
# OFFICIAL PAINT CONFIGURATION
# ---------------------------------------------------------

PAINT_MAX_NUM_VIEW = 9

PAINT_RESOLUTION = 768

# ---------------------------------------------------------
# OPTIONAL PERFORMANCE FEATURES
#
# Disabled by default.
#
# They are mainly for speed/memory, not visual quality.
# ---------------------------------------------------------

ENABLE_FLASHVDM = (
    os.environ.get(
        "HUNYUAN_ENABLE_FLASHVDM",
        "false"
    ).lower()
    in ("1", "true", "yes")
)

ENABLE_COMPILE = (
    os.environ.get(
        "HUNYUAN_ENABLE_COMPILE",
        "false"
    ).lower()
    in ("1", "true", "yes")
)


# =========================================================
# STARTUP INFORMATION
# =========================================================

print("")
print("======================================================")
print(" HUNYUAN3D-2.1 FINAL PRODUCTION WORKER")
print("======================================================")

print(
    "[INFO] Model:",
    MODEL_PATH
)

print(
    "[INFO] CUDA available:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required for Hunyuan3D-2.1."
    )

print(
    "[INFO] GPU:",
    torch.cuda.get_device_name(0)
)

gpu_memory = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / (1024 ** 3)
)

print(
    "[INFO] VRAM:",
    round(gpu_memory, 2),
    "GB"
)

print(
    "[INFO] Shape steps:",
    DEFAULT_STEPS
)

print(
    "[INFO] Shape octree:",
    DEFAULT_OCTREE_RESOLUTION
)

print(
    "[INFO] Paint views:",
    PAINT_MAX_NUM_VIEW
)

print(
    "[INFO] Paint resolution:",
    PAINT_RESOLUTION
)

print(
    "[INFO] FlashVDM:",
    ENABLE_FLASHVDM
)

print(
    "[INFO] Compile:",
    ENABLE_COMPILE
)


# =========================================================
# LOAD SHAPE MODEL
# =========================================================

print("")
print("[LOAD] Loading Hunyuan3D-2.1 shape model...")

shape_pipeline = (
    Hunyuan3DDiTFlowMatchingPipeline
    .from_pretrained(
        MODEL_PATH
    )
)

print(
    "[LOAD] Shape model ready."
)


# =========================================================
# OPTIONAL FLASHVDM
# =========================================================

if ENABLE_FLASHVDM:

    try:

        print(
            "[LOAD] Enabling FlashVDM..."
        )

        shape_pipeline.enable_flashvdm(
            mc_algo="mc"
        )

        print(
            "[LOAD] FlashVDM enabled."
        )

    except Exception as e:

        print(
            "[WARN] FlashVDM could not be enabled:",
            str(e)
        )

        print(
            "[WARN] Continuing with standard VAE decoding."
        )


# =========================================================
# OPTIONAL COMPILE
# =========================================================

if ENABLE_COMPILE:

    try:

        print(
            "[LOAD] Compiling shape pipeline..."
        )

        shape_pipeline.compile()

        print(
            "[LOAD] Shape pipeline compiled."
        )

    except Exception as e:

        print(
            "[WARN] Pipeline compilation failed:",
            str(e)
        )

        print(
            "[WARN] Continuing without compile."
        )


# =========================================================
# BACKGROUND REMOVER
# =========================================================

print("")
print(
    "[LOAD] Loading background remover..."
)

background_remover = BackgroundRemover()

print(
    "[LOAD] Background remover ready."
)


# =========================================================
# OFFICIAL PBR PAINT PIPELINE
# =========================================================

print("")
print(
    "[LOAD] Loading official Hunyuan3D-2.1 PBR pipeline..."
)

paint_config = Hunyuan3DPaintConfig(
    max_num_view=PAINT_MAX_NUM_VIEW,
    resolution=PAINT_RESOLUTION,
)

# ---------------------------------------------------------
# OFFICIAL PATHS
# ---------------------------------------------------------

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

print(
    "[LOAD] Official PBR paint pipeline ready."
)


# =========================================================
# READY
# =========================================================

print("")
print("======================================================")
print(" HUNYUAN3D-2.1 FINAL WORKER READY")
print("======================================================")
print("")


# =========================================================
# TIMER
# =========================================================

class Timer:

    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):

        return round(
            time.perf_counter() - self.start,
            3
        )


# =========================================================
# BASE64 DECODER
# =========================================================

def decode_image_base64(data):

    if not isinstance(
        data,
        str
    ):

        raise ValueError(
            "image_base64 must be a string."
        )

    data = data.strip()

    # -----------------------------------------------------
    # DATA URL
    # -----------------------------------------------------

    if data.startswith("data:"):

        if "," not in data:

            raise ValueError(
                "Invalid image data URL."
            )

        data = data.split(
            ",",
            1
        )[1]

    # -----------------------------------------------------
    # REMOVE WHITESPACE
    # -----------------------------------------------------

    data = "".join(
        data.split()
    )

    if not data:

        raise ValueError(
            "Empty image data."
        )

    # -----------------------------------------------------
    # DECODE
    # -----------------------------------------------------

    try:

        return base64.b64decode(
            data,
            validate=True
        )

    except Exception as e:

        raise ValueError(
            f"Invalid base64 image: {e}"
        )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    timer = Timer()

    print("")
    print("------------------------------------------------------")
    print(" IMAGE PREPROCESSING")
    print("------------------------------------------------------")

    print(
        "[IMAGE] Original:",
        image.size,
        image.mode
    )

    # -----------------------------------------------------
    # EXIF ORIENTATION
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
    # BACKGROUND REMOVAL
    # -----------------------------------------------------

    print(
        "[IMAGE] Removing background..."
    )

    rgba = background_remover(
        image
    )

    if rgba is None:

        raise RuntimeError(
            "Background remover returned None."
        )

    rgba = rgba.convert(
        "RGBA"
    )

    # -----------------------------------------------------
    # ALPHA
    #
    # IMPORTANT:
    # Do NOT aggressively blur the alpha.
    #
    # Hair, hats, fingers, shoes, weapons/accessories,
    # ears and other thin geometry can be damaged by
    # excessive alpha smoothing.
    # -----------------------------------------------------

    alpha = rgba.getchannel(
        "A"
    )

    alpha_np = np.asarray(
        alpha,
        dtype=np.uint8
    )

    # Only remove almost-invisible noise.
    alpha_np = np.where(
        alpha_np < 3,
        0,
        alpha_np
    ).astype(
        np.uint8
    )

    alpha = Image.fromarray(
        alpha_np,
        mode="L"
    )

    rgba.putalpha(
        alpha
    )

    # -----------------------------------------------------
    # FIND SUBJECT
    # -----------------------------------------------------

    bbox = alpha.getbbox()

    if bbox is None:

        raise RuntimeError(
            "No foreground subject detected."
        )

    left, top, right, bottom = bbox

    subject_width = (
        right - left
    )

    subject_height = (
        bottom - top
    )

    if (
        subject_width < 8
        or subject_height < 8
    ):

        raise RuntimeError(
            "Detected subject is too small."
        )

    print(
        "[IMAGE] Subject bbox:",
        bbox
    )

    # -----------------------------------------------------
    # CHARACTER PADDING
    #
    # Enough space for hair/accessories.
    # -----------------------------------------------------

    pad_x = max(
        12,
        int(
            subject_width * 0.08
        )
    )

    pad_y = max(
        12,
        int(
            subject_height * 0.08
        )
    )

    left = max(
        0,
        left - pad_x
    )

    top = max(
        0,
        top - pad_y
    )

    right = min(
        rgba.width,
        right + pad_x
    )

    bottom = min(
        rgba.height,
        bottom + pad_y
    )

    cropped = rgba.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    # -----------------------------------------------------
    # SQUARE CANVAS
    # -----------------------------------------------------

    width, height = cropped.size

    square_size = max(
        width,
        height
    )

    # 12% breathing room.
    square_size = int(
        square_size * 1.12
    )

    square_size = max(
        square_size,
        512
    )

    canvas = Image.new(
        "RGBA",
        (
            square_size,
            square_size
        ),
        (
            255,
            255,
            255,
            0
        )
    )

    x = (
        square_size - width
    ) // 2

    y = (
        square_size - height
    ) // 2

    canvas.alpha_composite(
        cropped,
        (
            x,
            y
        )
    )

    # -----------------------------------------------------
    # FINAL MODEL INPUT
    #
    # Hunyuan's image conditioning benefits from a clean
    # square input with consistent framing.
    # -----------------------------------------------------

    model_size = 1024

    model_input = canvas.resize(
        (
            model_size,
            model_size
        ),
        Image.Resampling.LANCZOS
    )

    print(
        "[IMAGE] Final input:",
        model_input.size
    )

    print(
        "[IMAGE] Preprocessing:",
        timer.elapsed(),
        "sec"
    )

    return model_input


# =========================================================
# MESH VALIDATION
# =========================================================

def validate_mesh(mesh):

    if mesh is None:

        raise RuntimeError(
            "Hunyuan returned no mesh."
        )

    try:

        vertices = np.asarray(
            mesh.vertices
        )

        faces = np.asarray(
            mesh.faces
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to read generated mesh: {e}"
        )

    # -----------------------------------------------------
    # Minimum geometry
    # -----------------------------------------------------

    if len(vertices) < 20:

        raise RuntimeError(
            "Generated mesh has too few vertices."
        )

    if len(faces) < 20:

        raise RuntimeError(
            "Generated mesh has too few faces."
        )

    # -----------------------------------------------------
    # FINITE VALUES
    # -----------------------------------------------------

    if not np.all(
        np.isfinite(vertices)
    ):

        raise RuntimeError(
            "Generated mesh contains NaN/Inf vertices."
        )

    # -----------------------------------------------------
    # BOUNDS
    # -----------------------------------------------------

    try:

        extents = np.asarray(
            mesh.extents,
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(extents)
        ):

            raise RuntimeError(
                "Generated mesh has invalid bounds."
            )

        if np.max(extents) <= 0:

            raise RuntimeError(
                "Generated mesh has zero size."
            )

    except RuntimeError:

        raise

    except Exception as e:

        raise RuntimeError(
            f"Mesh bounds validation failed: {e}"
        )

    return True


# =========================================================
# SAFE MESH CLEANUP
# =========================================================

def clean_mesh(mesh):

    """
    Conservative cleanup.

    We intentionally DO NOT remove small connected
    components because characters may legitimately contain:

        eyes
        hair pieces
        shoes
        accessories
        fingers
        clothing parts

    The official PBR pipeline also performs its own mesh
    processing/remeshing.
    """

    timer = Timer()

    print("")
    print("------------------------------------------------------")
    print(" MESH VALIDATION")
    print("------------------------------------------------------")

    validate_mesh(
        mesh
    )

    # -----------------------------------------------------
    # Remove duplicate vertices
    # -----------------------------------------------------

    try:

        mesh.remove_duplicate_vertices()

    except Exception as e:

        print(
            "[MESH] Duplicate vertex cleanup skipped:",
            str(e)
        )

    # -----------------------------------------------------
    # Remove unreferenced vertices
    # -----------------------------------------------------

    try:

        mesh.remove_unreferenced_vertices()

    except Exception as e:

        print(
            "[MESH] Unreferenced vertex cleanup skipped:",
            str(e)
        )

    # -----------------------------------------------------
    # Remove degenerate faces
    # -----------------------------------------------------

    try:

        mesh.remove_degenerate_faces()

    except Exception as e:

        print(
            "[MESH] Degenerate face cleanup skipped:",
            str(e)
        )

    # -----------------------------------------------------
    # Remove infinite values
    # -----------------------------------------------------

    try:

        mesh.remove_infinite_values()

    except Exception as e:

        print(
            "[MESH] Infinite-value cleanup skipped:",
            str(e)
        )

    # -----------------------------------------------------
    # Repair normals
    # -----------------------------------------------------

    try:

        mesh.fix_normals()

    except Exception as e:

        print(
            "[MESH] Normal repair skipped:",
            str(e)
        )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    validate_mesh(
        mesh
    )

    print(
        "[MESH] Vertices:",
        len(mesh.vertices)
    )

    print(
        "[MESH] Faces:",
        len(mesh.faces)
    )

    print(
        "[MESH] Validation:",
        timer.elapsed(),
        "sec"
    )

    return mesh


# =========================================================
# SHAPE GENERATION
# =========================================================

def generate_shape(
    image,
    seed,
    steps,
    octree_resolution,
    guidance_scale,
    num_chunks
):

    timer = Timer()

    print("")
    print("======================================================")
    print(" HUNYUAN3D SHAPE GENERATION")
    print("======================================================")

    print(
        "[SHAPE] Seed:",
        seed
    )

    print(
        "[SHAPE] Steps:",
        steps
    )

    print(
        "[SHAPE] Octree:",
        octree_resolution
    )

    print(
        "[SHAPE] Guidance:",
        guidance_scale
    )

    print(
        "[SHAPE] Chunks:",
        num_chunks
    )

    # -----------------------------------------------------
    # RANDOM GENERATOR
    # -----------------------------------------------------

    generator = torch.Generator(
        device=shape_pipeline.device
    )

    generator.manual_seed(
        seed
    )

    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------

    with torch.inference_mode():

        result = shape_pipeline(

            image=image,

            num_inference_steps=steps,

            guidance_scale=guidance_scale,

            octree_resolution=octree_resolution,

            num_chunks=num_chunks,

            generator=generator,

            mc_algo="mc",

            output_type="trimesh",

            enable_pbar=False,
        )

    if result is None:

        raise RuntimeError(
            "Shape pipeline returned None."
        )

    if len(result) == 0:

        raise RuntimeError(
            "Shape pipeline returned an empty result."
        )

    mesh = result[0]

    validate_mesh(
        mesh
    )

    print(
        "[SHAPE] Generation time:",
        timer.elapsed(),
        "sec"
    )

    return mesh


# =========================================================
# SAVE MESH
# =========================================================

def save_mesh(mesh):

    file = tempfile.NamedTemporaryFile(
        suffix=".glb",
        delete=False
    )

    path = file.name

    file.close()

    try:

        mesh.export(
            path
        )

    except Exception:

        try:
            os.remove(path)
        except Exception:
            pass

        raise

    if not os.path.exists(path):

        raise RuntimeError(
            "Failed to create temporary GLB."
        )

    if os.path.getsize(path) == 0:

        raise RuntimeError(
            "Temporary GLB is empty."
        )

    return path


# =========================================================
# PBR TEXTURE GENERATION
# =========================================================

def generate_texture(
    mesh_path,
    image_path
):

    timer = Timer()

    print("")
    print("======================================================")
    print(" OFFICIAL HUNYUAN PBR TEXTURE GENERATION")
    print("======================================================")

    print(
        "[PAINT] Views:",
        PAINT_MAX_NUM_VIEW
    )

    print(
        "[PAINT] Resolution:",
        PAINT_RESOLUTION
    )

    output_file = tempfile.NamedTemporaryFile(
        suffix="_textured.glb",
        delete=False
    )

    output_path = output_file.name

    output_file.close()

    try:

        with torch.inference_mode():

            result = paint_pipeline(

                mesh_path=mesh_path,

                image_path=image_path,

                output_mesh_path=output_path
            )

    except Exception:

        try:

            os.remove(output_path)

        except Exception:
            pass

        raise

    # -----------------------------------------------------
    # PIPELINE MAY RETURN OUTPUT PATH
    # -----------------------------------------------------

    final_path = result

    if not final_path:

        final_path = output_path

    if not isinstance(
        final_path,
        str
    ):

        final_path = output_path

    if not os.path.exists(
        final_path
    ):

        raise RuntimeError(
            "PBR pipeline did not create a GLB."
        )

    if os.path.getsize(
        final_path
    ) == 0:

        raise RuntimeError(
            "PBR pipeline created an empty GLB."
        )

    print(
        "[PAINT] Generation time:",
        timer.elapsed(),
        "sec"
    )

    return final_path


# =========================================================
# CLEANUP HELPER
# =========================================================

def remove_file(path):

    if not path:

        return

    try:

        if os.path.exists(path):

            os.remove(path)

    except Exception as e:

        print(
            "[CLEANUP] Could not remove:",
            path,
            str(e)
        )


# =========================================================
# CUDA CLEANUP
# =========================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        try:

            torch.cuda.empty_cache()

        except Exception:
            pass

        try:

            torch.cuda.ipc_collect()

        except Exception:
            pass


# =========================================================
# INPUT VALIDATION
# =========================================================

def get_integer(
    value,
    name,
    minimum,
    maximum
):

    try:

        value = int(value)

    except Exception:

        raise ValueError(
            f"{name} must be an integer."
        )

    if value < minimum:

        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    if value > maximum:

        raise ValueError(
            f"{name} must be <= {maximum}."
        )

    return value


def get_float(
    value,
    name,
    minimum,
    maximum
):

    try:

        value = float(value)

    except Exception:

        raise ValueError(
            f"{name} must be a number."
        )

    if value < minimum:

        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    if value > maximum:

        raise ValueError(
            f"{name} must be <= {maximum}."
        )

    return value


# =========================================================
# MAIN RUNPOD HANDLER
# =========================================================

def handler(job):

    start_time = time.perf_counter()

    image_path = None

    shape_path = None

    textured_path = None

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
                "job.input must be an object."
            )

        image_b64 = job_input.get(
            "image_base64"
        )

        if not image_b64:

            raise ValueError(
                "Missing image_base64."
            )

        # =================================================
        # SETTINGS
        # =================================================

        seed = get_integer(

            job_input.get(
                "seed",
                DEFAULT_SEED
            ),

            "seed",

            0,

            2**32 - 1
        )

        steps = get_integer(

            job_input.get(
                "steps",
                DEFAULT_STEPS
            ),

            "steps",

            20,

            100
        )

        octree_resolution = get_integer(

            job_input.get(
                "octree_resolution",
                DEFAULT_OCTREE_RESOLUTION
            ),

            "octree_resolution",

            128,

            512
        )

        guidance_scale = get_float(

            job_input.get(
                "guidance_scale",
                DEFAULT_GUIDANCE_SCALE
            ),

            "guidance_scale",

            1.0,

            10.0
        )

        num_chunks = get_integer(

            job_input.get(
                "num_chunks",
                DEFAULT_NUM_CHUNKS
            ),

            "num_chunks",

            2000,

            20000
        )

        # =================================================
        # LOG
        # =================================================

        print("")
        print("")
        print("======================================================")
        print(" NEW HUNYUAN3D GENERATION")
        print("======================================================")

        print(
            "[CONFIG] Seed:",
            seed
        )

        print(
            "[CONFIG] Steps:",
            steps
        )

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

        print(
            "[CONFIG] Paint views:",
            PAINT_MAX_NUM_VIEW
        )

        print(
            "[CONFIG] Paint resolution:",
            PAINT_RESOLUTION
        )

        # =================================================
        # DECODE IMAGE
        # =================================================

        print("")
        print(
            "[IMAGE] Decoding..."
        )

        image_bytes = decode_image_base64(
            image_b64
        )

        try:

            original_image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            original_image.load()

        except Exception as e:

            raise ValueError(
                f"Unable to open uploaded image: {e}"
            )

        print(
            "[IMAGE] Uploaded:",
            original_image.size,
            original_image.mode
        )

        # =================================================
        # PREPROCESS
        # =================================================

        model_image = preprocess_image(
            original_image
        )

        # =================================================
        # SAVE INPUT IMAGE
        # =================================================

        image_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        image_path = image_file.name

        image_file.close()

        model_image.save(
            image_path,
            format="PNG"
        )

        print(
            "[IMAGE] Prepared image:",
            image_path
        )

        # =================================================
        # SHAPE
        # =================================================

        mesh = generate_shape(

            image=model_image,

            seed=seed,

            steps=steps,

            octree_resolution=octree_resolution,

            guidance_scale=guidance_scale,

            num_chunks=num_chunks
        )

        # =================================================
        # SAFE CLEANUP
        # =================================================

        mesh = clean_mesh(
            mesh
        )

        # =================================================
        # SAVE SHAPE
        # =================================================

        shape_path = save_mesh(
            mesh
        )

        print(
            "[MESH] Shape GLB:",
            shape_path
        )

        # =================================================
        # TEXTURE
        # =================================================

        textured_path = generate_texture(

            mesh_path=shape_path,

            image_path=image_path
        )

        # =================================================
        # READ GLB
        # =================================================

        with open(
            textured_path,
            "rb"
        ) as f:

            output_bytes = f.read()

        if not output_bytes:

            raise RuntimeError(
                "Final GLB is empty."
            )

        # =================================================
        # BASE64
        # =================================================

        output_b64 = base64.b64encode(
            output_bytes
        ).decode(
            "utf-8"
        )

        # =================================================
        # TOTAL TIME
        # =================================================

        total_time = round(
            time.perf_counter()
            - start_time,
            3
        )

        print("")
        print("======================================================")
        print(" GENERATION SUCCESS")
        print("======================================================")

        print(
            "[OUTPUT] Size:",
            round(
                len(output_bytes) / (1024 * 1024),
                2
            ),
            "MB"
        )

        print(
            "[OUTPUT] Total time:",
            total_time,
            "seconds"
        )

        return {

            "model_base64":
                output_b64,

            "format":
                "glb",

            "success":
                True,

            "settings": {

                "seed":
                    seed,

                "steps":
                    steps,

                "octree_resolution":
                    octree_resolution,

                "guidance_scale":
                    guidance_scale,

                "num_chunks":
                    num_chunks,

                "texture_views":
                    PAINT_MAX_NUM_VIEW,

                "texture_resolution":
                    PAINT_RESOLUTION,

                "flashvdm":
                    ENABLE_FLASHVDM,

                "compile":
                    ENABLE_COMPILE,

                "generation_time_seconds":
                    total_time
            }
        }

    except Exception as e:

        print("")
        print("======================================================")
        print(" GENERATION FAILED")
        print("======================================================")

        print(
            "[ERROR]",
            str(e)
        )

        traceback.print_exc()

        return {

            "success":
                False,

            "error":
                str(e),

            "type":
                type(e).__name__
        }

    finally:

        print(
            "[CLEANUP] Cleaning temporary files..."
        )

        remove_file(
            image_path
        )

        remove_file(
            shape_path
        )

        remove_file(
            textured_path
        )

        cleanup_memory()

        print(
            "[CLEANUP] Complete."
        )


# =========================================================
# RUNPOD SERVERLESS
# =========================================================

print(
    "[RUNPOD] Starting serverless worker..."
)

runpod.serverless.start({
    "handler": handler
})
