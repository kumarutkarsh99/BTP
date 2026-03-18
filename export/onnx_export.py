"""
ONNX Export Module
"""
import torch
import logging

logger = logging.getLogger(__name__)


def export_to_onnx(model, output_path, sample_input=None, dynamic_batch=True):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        sample_input: Sample input dict (if None, will create dummy)
        dynamic_batch: Support dynamic batch sizes
    
    Returns:
        bool: Success
    """
    try:
        model.eval()
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = {
                'input_ids': torch.randint(0, 1000, (1, 128)),
                'attention_mask': torch.ones(1, 128, dtype=torch.long),
            }
        
        # Move to CPU for export
        model = model.cpu()
        sample_input = {k: v.cpu() for k, v in sample_input.items()}
        
        # Dynamic axes
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes = {
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            }
        
        # Export
        torch.onnx.export(
            model,
            (sample_input,),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
        )
        
        logger.info(f"✅ Model exported to {output_path}")
        
        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return False


def benchmark_onnx(onnx_path, pytorch_model, test_loader, device='cpu'):
    """
    Compare ONNX vs PyTorch inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        test_loader: Test dataloader
        device: Device for comparison
    
    Returns:
        dict: Benchmark results
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import time
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        pytorch_times = []
        onnx_times = []
        
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(device)
        
        for i, batch in enumerate(test_loader):
            if i >= 100:  # Test on 100 batches
                break
            
            input_ids = batch['input_ids'].numpy()
            attention_mask = batch['attention_mask'].numpy()
            
            # PyTorch inference
            start = time.time()
            with torch.no_grad():
                pytorch_out = pytorch_model(
                    input_ids=torch.from_numpy(input_ids).to(device),
                    attention_mask=torch.from_numpy(attention_mask).to(device)
                )
            pytorch_times.append(time.time() - start)
            
            # ONNX inference
            start = time.time()
            onnx_out = ort_session.run(
                None,
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            )
            onnx_times.append(time.time() - start)
        
        results = {
            'pytorch_avg_ms': np.mean(pytorch_times) * 1000,
            'onnx_avg_ms': np.mean(onnx_times) * 1000,
            'speedup': np.mean(pytorch_times) / np.mean(onnx_times),
        }
        
        logger.info(f"PyTorch: {results['pytorch_avg_ms']:.2f}ms")
        logger.info(f"ONNX: {results['onnx_avg_ms']:.2f}ms")
        logger.info(f"Speedup: {results['speedup']:.2f}×")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return None