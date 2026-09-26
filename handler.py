"""
RunPod Serverless Handler for Hunyuan3D-2.1
---------------------------------------------
This file is the "connector" between RunPod's serverless system and
Tencent's Hunyuan3D-2.1 code. Place this file at the ROOT of your forked
repo (gulambahoogb-coder/Hunyuan3D-2.1), alongside api_server.py.

WHAT THIS DOES:
1. Loads the Hunyuan3D-2.1 model ONE TIME when the server starts
   (not on every request — that would be too slow).
2. Defines a function that RunPod calls every time your website
   sends a request.
3. That function takes an image, runs it through the model, and
   returns a 3D model file (as base64 text, since RunPod jobs
   return JSON, not raw files).
"""

import sys
import os
import base64
import tempfile

# Make sure Python can find Tencent's code folders
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import runpod
from PIL import Image
import io

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print(f"torchvision_fix skipped: {e}")

# -----------------------------
# STEP 1: Load the model ONCE
# -----------------------------
print("Loading Hunyuan3D-2.1 shape model... (this happens once at startup)")
MODEL_PATH = "/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1"

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH)

paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
paint_config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
paint_config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
paint_config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

background_remover = BackgroundRemover()

print("Model loaded. Ready to accept requests.")


# -----------------------------------------
# STEP 2: The function RunPod will call
# -----------------------------------------
def handler(job):
    """
    'job' is what RunPod passes in when your website sends a request.
    Expected input format (sent from your backend):
    {
        "input": {
            "image_base64": "<base64-encoded image string>"
        }
    }
    """
    try:
        job_input = job["input"]
        image_b64 = job_input["image_base64"]

        # Decode the incoming image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # Remove background if needed (model works best on clean subjects)
        if image.mode == "RGB":
            image = background_remover(image)

        # STEP A: Generate the 3D shape (no texture yet)
        mesh = shape_pipeline(image=image)[0]

        # Save shape to a temporary file so the paint step can use it
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_shape:
            mesh.export(tmp_shape.name)
            shape_path = tmp_shape.name

        # Save the input image temporarily too (paint step needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            image.save(tmp_img.name)
            image_path = tmp_img.name

        # STEP B: Add realistic texture/color to the shape
        textured_output_path = tempfile.NamedTemporaryFile(suffix="_textured.glb", delete=False).name
        textured_mesh_path = paint_pipeline(
            mesh_path=shape_path,
            image_path=image_path,
            output_mesh_path=textured_output_path
        )

        # Read the final textured model and encode it to send back
        with open(textured_mesh_path, "rb") as f:
            result_bytes = f.read()
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Clean up temp files
        os.remove(shape_path)
        os.remove(image_path)

        return {
            "model_base64": result_b64,
            "format": "glb"
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------------------
# STEP 3: Register the handler with RunPod
# -----------------------------------------
runpod.serverless.start({"handler": handler})
