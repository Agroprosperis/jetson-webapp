import os
import glob
import subprocess
import sys
import shutil
import argparse

# 1. Ensure rfdetr is installed
try:
    import rfdetr
    print("rfdetr package found.")
except ImportError:
    print("rfdetr package not found. Installing via pip (runtime)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rfdetr"])
    import rfdetr

# 2. Ensure tensorrt is importable
# Roboflow Runtime images often lack the python bindings, even if the libs are there.
try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("TensorRT python package not found. Attempting to install via pip...")
    try:
        # This works for Desktop x86_64 environments
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorrt"])
        import tensorrt as trt
        print(f"SUCCESS: Installed TensorRT {trt.__version__}")
    except Exception as e:
        print("\nCRITICAL ERROR: Could not install 'tensorrt' python package.")
        print("Desktop setup requires network access to PyPI to install TensorRT.")
        print(f"Error details: {e}")
        sys.exit(1)

from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRSegPreview

def build_engine_python(onnx_path, engine_path, fp16=True):
    """
    Compiles an ONNX model to TensorRT Engine using the native Python API.
    """
    print(f"Building TensorRT Engine via Python API for: {onnx_path}")
    
    # Mute the verbose logger slightly to reduce noise
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH flag is mandatory for ONNX models
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Enable FP16 if requested and supported by hardware
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Memory pool limit (2GB) for optimization tactics
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file {onnx_path} not found.")
        return False

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("Building serialized network (this may take a few minutes)...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Error: build_serialized_network returned None.")
            return False
            
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
            
        print(f"Engine successfully saved to: {engine_path}")
        return True
        
    except Exception as e:
        print(f"Exception during engine build: {e}")
        return False

def load_and_export(pt_file, output_path):
    print(f"\n---------------------------------------------------")
    print(f"Processing: {pt_file}")
    
    model_name = os.path.basename(pt_file).lower()
    
    try:
        # A. Instantiate the correct model class
        if "seg" in model_name:
            print(f"Detected Segmentation model: {model_name}")
            model = RFDETRSegPreview(pretrain_weights=pt_file)
        elif "nano" in model_name:
            model = RFDETRNano(pretrain_weights=pt_file, resolution=640)
        elif "small" in model_name:
            model = RFDETRSmall(pretrain_weights=pt_file)
        elif "medium" in model_name:
            model = RFDETRMedium(pretrain_weights=pt_file)
        elif "large" in model_name:
            model = RFDETRLarge(pretrain_weights=pt_file)
        else:
            print("No size detected in filename, defaulting to RFDETRBase")
            model = RFDETRBase(pretrain_weights=pt_file)
            
        print(f"Model loaded. Exporting to ONNX intermediate...")
        
        # B. Export to ONNX only (we handle TRT build ourselves)
        onnx_export_path = model.export(
            format="onnx", 
            simplify=True
        )
        
        # C. Robustly locate the generated ONNX file
        if not onnx_export_path or not os.path.exists(onnx_export_path):
            candidates = [
                os.path.join(os.getcwd(), "output", "inference_model.sim.onnx"),
                os.path.join(os.getcwd(), "output", "inference_model.onnx"),
                pt_file.replace(".pt", ".onnx")
            ]
            for c in candidates:
                if os.path.exists(c):
                    onnx_export_path = c
                    break
        
        if not onnx_export_path or not os.path.exists(onnx_export_path):
            print("CRITICAL: ONNX export failed. Cannot proceed to TensorRT build.")
            return

        print(f"ONNX intermediate found at: {onnx_export_path}")

        # D. Build TensorRT Engine
        success = build_engine_python(onnx_export_path, output_path, fp16=True)
        
        if success:
            print(f"SUCCESS: Exported {pt_file} -> {output_path}")
        else:
            print(f"FAILED to build engine for {pt_file}")

        # E. Cleanup
        if os.path.exists("output"):
            try:
                shutil.rmtree("output")
            except:
                pass

    except Exception as e:
        print(f"FAILED to convert {pt_file}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to a single .pt model to compile")
    args = parser.parse_args()

    model_dir = "/app/model/rf"
    if args.model:
        model_dir = os.path.dirname(args.model) or model_dir

    # Safety check for mount
    if not os.path.exists(model_dir):
        print(f"ERROR: {model_dir} does not exist.")
        print("Did you forget to mount your models? (-v /local/path:/app/model/rf)")
        sys.exit(1)

    if args.model:
        if not os.path.exists(args.model):
            print(f"ERROR: {args.model} does not exist.")
            sys.exit(1)
        files = [args.model]
    else:
        files = glob.glob(os.path.join(model_dir, "*.pt"))

    if not files:
        print(f"No .pt files found in {model_dir}")
        return

    print(f"Found {len(files)} models.")

    for pt_file in files:
        base = os.path.splitext(os.path.basename(pt_file))[0]
        out_name = os.path.join(os.path.dirname(pt_file), f"{base}-fp16.engine")

        if os.path.exists(out_name):
            print(f"Skipping {out_name} (already exists)")
            continue

        load_and_export(pt_file, out_name)

if __name__ == "__main__":
    main()
