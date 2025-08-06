#!/usr/bin/env python3
"""
Final ONNX conversion script for YOLOv5 best.pt
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def convert_to_onnx():
    # Path to the best.pt model
    weights_path = Path("runs/train/yolov5s_test22/weights/best.pt")
    onnx_path = weights_path.parent / "best.onnx"
    
    if not weights_path.exists():
        print(f"Error: Model file not found at {weights_path}")
        return
    
    print(f"Loading model from {weights_path}...")
    
    try:
        # Load the model using torch.load
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Extract the model
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
        
        # Convert model to full precision (FP32) and move to CPU
        model = model.float().cpu()
        
        # Set model to evaluation mode
        model.eval()
        
        # Disable training mode for all modules
        for module in model.modules():
            if hasattr(module, 'training'):
                module.training = False
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 416, 416, dtype=torch.float32)
        
        print(f"Converting to ONNX: {onnx_path}")
        
        # Export to ONNX with error handling
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes=None,
                verbose=False
            )
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        
        # Print file size
        file_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"📁 ONNX file size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_to_onnx() 