import torch
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import quantizer # Import our other file
import time
import torch.nn.utils.prune as prune
import pickle
import gzip
import numpy as np

# --- 1. Evaluation Functions ---

@torch.no_grad()
def get_all_predictions(model, dataloader, device):
    """
    Runs a model over a test dataloader and returns all predictions and labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        if isinstance(batch, list):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            model_input = data 
        elif isinstance(batch, dict):
            # Handle HuggingFace dicts
            labels = batch.pop('label', None)
            if labels is None and 'labels' in batch: # Sometimes it's plural
                labels = batch.pop('labels')
            
            if labels is not None:
                labels = labels.to(device)
            
            model_input = {k: v.to(device) for k, v in batch.items()}
        else:
            raise TypeError(f"Unknown batch type: {type(batch)}")
        
        # Forward pass
        if isinstance(model_input, dict):
            outputs = model(**model_input)
        else:
            outputs = model(model_input)
        
        # Handle HuggingFace outputs (logits) or raw tensors
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        pred = torch.argmax(logits, dim=1)
            
        all_preds.extend(pred.cpu().numpy())
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())
        
    return all_labels, all_preds

@torch.no_grad()
def get_accuracy(model, dataloader, device):
    """Helper function to get just the accuracy number."""
    y_true, y_pred = get_all_predictions(model, dataloader, device)
    if not y_true: return 0.0 # Handle case with no labels
    return metrics.accuracy_score(y_true, y_pred)

@torch.no_grad()
def calculate_perplexity(model, dataloader, device):
    """
    Calculates Perplexity (PPL) for classification/language tasks.
    PPL = exp(CrossEntropyLoss). 
    Lower is better.
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    loss_fct = nn.CrossEntropyLoss()

    for batch in dataloader:
        if isinstance(batch, dict):
            labels = batch.pop('label', None)
            if labels is None and 'labels' in batch: labels = batch.pop('labels')
            labels = labels.to(device)
            
            model_input = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**model_input)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            logits = outputs
        
        loss = loss_fct(logits, labels)
        total_loss += loss.item()
        total_steps += 1

    if total_steps == 0: return 0.0
    avg_loss = total_loss / total_steps
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

def plot_confusion_matrix(y_true, y_pred, labels, title):
    """Plots a confusion matrix and prints a full classification report."""
    acc = metrics.accuracy_score(y_true, y_pred)
    title_with_acc = f"{title}\n(Acc: {acc*100:.2f}%)"
    
    print(f"\n--- Classification Report for: {title} ---")
    print(metrics.classification_report(y_true, y_pred, target_names=[str(l) for l in labels]))
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(title_with_acc)
    plt.show()

# --- 2. Quantization Application Function ---

@torch.no_grad()
def apply_quantization_to_model(model, quant_func_or_profile):
    """
    Applies quantization to a model.
    Handles uniform (function) or mixed-precision (dict profile).
    Note: For 'search' phase, we apply this temporarily.
    """
    quant_map = {
        'INT4': quantizer.quant_af4_func,
        'INT8': quantizer.quant_af8_func,
        'FP32': lambda x: x 
    }

    if callable(quant_func_or_profile):
        quant_func = quant_func_or_profile
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1: # Only quantize matrices
                param.data.copy_(quant_func(param.data))
    
    elif isinstance(quant_func_or_profile, dict):
        profile = quant_func_or_profile
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                precision = profile.get(name, 'FP32') 
                if precision in quant_map:
                    quant_func = quant_map[precision]
                    param.data.copy_(quant_func(param.data))
                else:
                    pass 

# --- 3. STRUCTURED PRUNING FUNCTIONS (UPDATED) ---

def apply_structured_pruning(model, amount=0.3):
    """
    Applies STRUCTURED pruning (removing entire channels/rows).
    This physically allows for file size reduction without heavy index overhead.
    """
    print(f"✂️ Applying {amount*100}% STRUCTURED Pruning (L2 Norm)...")
    
    parameters_to_prune = []
    # We target Linear layers (Dense matrices)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Prune 'dim=0' (Outputs/Rows). This corresponds to removing neurons.
    # We use n=2 (L2 Norm) to find the "weakest" rows.
    for module, name in parameters_to_prune:
        prune.ln_structured(
            module, 
            name=name, 
            amount=amount, 
            n=2,       # L2 Norm (Standard for structured)
            dim=0      # 0 = Prune Rows (Output Channels)
        )
        
    print(f"   Structured Pruning applied to {len(parameters_to_prune)} layers.")

def make_pruning_permanent(model):
    """Removes the mask and permanently sets weights to zero."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    print("   Pruning masks removed (zeros made permanent).")

# --- 4. Model Size, Sparsity & Hardware Stats ---

def get_model_size_mb(model):
    """Calculates model size in Megabytes (FP32 assumption)."""
    return sum(p.numel() for p in model.parameters()) * 4 / (1024**2) 

def get_model_sparsity(model):
    """Calculates the percentage of zeroed weights in a model."""
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    return (zero_params / total_params) * 100 if total_params > 0 else 0

def check_hardware_compatibility(model, device):
    """
    Checks if the model fits in device memory.
    """
    model_size_mb = get_model_size_mb(model)
    stats = {
        "model_size_mb": model_size_mb,
        "device": str(device),
        "fits_in_memory": True,
        "dtypes_used": set()
    }
    
    # Collect utilized dtypes
    for param in model.parameters():
        stats["dtypes_used"].add(str(param.dtype))
        
    # Check VRAM if using CUDA
    if device.type == 'cuda':
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            free_mem_mb = free_mem / (1024**2)
            stats["free_vram_mb"] = free_mem_mb
            if model_size_mb > free_mem_mb:
                stats["fits_in_memory"] = False
        except Exception as e:
            stats["memory_check_error"] = str(e)
            
    return stats

# --- 5. Computational Cost & Efficiency (FLOPs/Energy) ---

def count_flops(model, input_sample):
    """
    Estimates FLOPs (Floating Point Operations) for Linear and Conv2d layers.
    Note: MACs (Multiply-Accumulates) * 2 = FLOPs approx.
    """
    total_macs = 0
    
    def hook_fn(module, input, output):
        nonlocal total_macs
        if isinstance(module, nn.Linear):
            # MACs = in_features * out_features
            total_macs += module.in_features * module.out_features
        if isinstance(module, nn.Conv2d):
            # MACs = k*k * in_channels * out_channels * out_h * out_w
            out_h, out_w = output.shape[2], output.shape[3]
            total_macs += (module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels * out_h * out_w)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a dummy pass
    if isinstance(input_sample, dict):
        model(**input_sample)
    else:
        model(input_sample)
        
    for h in hooks:
        h.remove()
        
    return total_macs * 2  # Convert MACs to FLOPs

def estimate_energy_efficiency(flops, avg_bit_width=32):
    """
    Estimates energy in Joules based on theoretical operations.
    Constants ref: 45nm CMOS technology (Horowitz et al.)
    FP32 add: 0.9 pJ, mult: 3.7 pJ -> ~4.6 pJ per MAC
    INT8 add: 0.03 pJ, mult: 0.2 pJ -> ~0.23 pJ per MAC
    """
    # Simple linear interpolation for custom bit widths (e.g., 4-bit)
    if avg_bit_width > 16:
        energy_per_flop_pj = 4.6 
    elif avg_bit_width > 4:
        energy_per_flop_pj = 0.23 # INT8
    else:
        energy_per_flop_pj = 0.12 # INT4 estimate
    
    total_energy_pj = flops * energy_per_flop_pj # picojoules
    total_energy_joules = total_energy_pj * 1e-12
    
    return total_energy_joules

# --- 6. Latency & Throughput Metrics ---

@torch.no_grad()
def measure_inference_speed(model, dataloader, device):
    """Measures latency and throughput."""
    model.eval()
    
    # Warmup
    print("Warming up for speed test...")
    iterator = iter(dataloader)
    try:
        for _ in range(min(5, len(dataloader))):
            batch = next(iterator)
            if isinstance(batch, dict):
                labels = batch.pop('label', None)
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                data, _ = batch
                model(data.to(device))
    except StopIteration:
        pass
    
    latencies = []
    total_samples = 0
    
    print("Measuring inference speed...")
    start_time = time.time()
    
    # We only run a subset for speed if dataloader is huge
    max_batches = 100 
    
    for i, batch in enumerate(tqdm(dataloader, desc="Benchmarking Speed")):
        if i >= max_batches: break
        
        batch_start = time.time()
        
        if isinstance(batch, dict):
            labels = batch.pop('label', None)
            if labels is None and 'labels' in batch: labels = batch.pop('labels')
            
            model_input = {k: v.to(device) for k, v in batch.items()}
            model(**model_input)
            
            # Robust batch size check
            current_batch_size = list(model_input.values())[0].size(0)
        else:
            data, labels = batch
            model(data.to(device))
            current_batch_size = labels.size(0)

        latencies.append(time.time() - batch_start)
        total_samples += current_batch_size
        
    total_time = time.time() - start_time
    
    avg_latency_ms = (np.mean(latencies) * 1000)
    throughput = total_samples / total_time
    
    return avg_latency_ms, throughput

# --- 7. CUSTOM SAVER & LOADER (The Core of Your Paper) ---

def pack_int4(int8_tensor):
    """
    Packs two INT4 values (stored as int8) into one byte (uint8).
    Range: -8 to 7.
    """
    # 1. Shift range: -8..7 -> 0..15
    data = int8_tensor.astype(np.uint8) + 8
    
    # 2. Split into high/low nibbles
    # Assuming even length. If odd, pad one zero.
    if len(data) % 2 != 0:
        data = np.append(data, 0)
    
    high_nibble = data[0::2] << 4
    low_nibble = data[1::2] & 0x0F
    
    packed = high_nibble | low_nibble
    return packed

def unpack_int4(packed_tensor, original_length):
    """
    Unpacks uint8 containers back into INT4 values (-8 to 7).
    """
    # 1. Unpack High/Low Nibbles
    high_nibble = (packed_tensor >> 4) & 0x0F
    low_nibble = packed_tensor & 0x0F
    
    # 2. Interleave them back into a single array
    unpacked = np.empty(len(packed_tensor) * 2, dtype=np.int8)
    unpacked[0::2] = high_nibble
    unpacked[1::2] = low_nibble
    
    # 3. Trim padding if original length was odd
    unpacked = unpacked[:original_length]
    
    # 4. Shift back range (0..15 -> -8..7)
    return unpacked.astype(np.int8) - 8

def save_compressed_model(model, profile, filename):
    """
    Saves the model in a custom 'Structured-Sparse' format.
    1. Removes zero-rows physically.
    2. Quantizes remaining weights to INT4/INT8 based on profile.
    3. Saves only the 'Active Rows' and the 'Row Map'.
    """
    print(f"📦 Structured Packing and Zipping to {filename}...")
    
    compressed_dict = {'weights': {}}
    
    for name, param in model.named_parameters():
        # Only compress weights, leave biases/embeddings as FP32/INT8 for safety
        if 'weight' in name and param.dim() > 1:
            # 1. Identify Active Rows (Structured Pruning Check)
            # Sum absolute values across columns. If sum == 0, row is dead.
            row_magnitudes = torch.sum(torch.abs(param.data), dim=1)
            active_indices = torch.nonzero(row_magnitudes)[:, 0].cpu().numpy()
            
            # 2. Extract Active Rows Only
            active_rows = param.data[active_indices, :].cpu().numpy()
            
            # 3. Quantize (Simulated -> Integer Storage)
            precision = profile.get(name, 'INT8') # Default to INT8 if not found
            
            if precision == 'INT4':
                # Map float to int range -8..7
                scale = np.max(np.abs(active_rows)) / 7.0
                if scale == 0: scale = 1.0
                int_data = np.round(active_rows / scale).astype(np.int8)
                packed_data = pack_int4(int_data.flatten())
                fmt = 'int4_packed'
            else:
                # INT8
                scale = np.max(np.abs(active_rows)) / 127.0
                if scale == 0: scale = 1.0
                packed_data = np.round(active_rows / scale).astype(np.int8).flatten()
                fmt = 'int8_flat'

            compressed_dict['weights'][name] = {
                'data': packed_data,
                'row_map': active_indices.astype(np.uint16), # Save indices of kept rows
                'original_shape': param.shape,
                'format': fmt,
                'scale': scale
            }
        else:
            # Save biases/others normally (or as FP16)
            compressed_dict['weights'][name] = param.data.cpu().numpy().astype(np.float32)

    # 4. Gzip Save
    if not filename.endswith(".gz"): filename += ".gz"
    with gzip.open(filename, 'wb') as f:
        pickle.dump(compressed_dict, f)
        
    print(f"   File saved: {filename}")

def load_compressed_model(model, filename, device='cpu'):
    """
    Loads a structured-compressed model (.pkl.gz) into a PyTorch model skeleton.
    Reconstructs the zeroed rows using the 'row_map'.
    """
    print(f"Loading compressed model from {filename}...")
    
    if not filename.endswith(".gz"): filename += ".gz"
    
    with gzip.open(filename, 'rb') as f:
        compressed_dict = pickle.load(f)
    
    # Iterate through model parameters and inject weights
    for name, param in model.named_parameters():
        if name not in compressed_dict['weights']:
            continue
            
        w_info = compressed_dict['weights'][name]
        
        # Case A: Standard Parameter (Bias, LayerNorm, etc.)
        if isinstance(w_info, np.ndarray):
            param.data = torch.from_numpy(w_info).to(device).type(param.dtype)
            continue
            
        # Case B: Structured Compressed Matrix
        # 1. Prepare Empty Tensor (Original Shape)
        full_shape = w_info['original_shape']
        reconstructed_tensor = torch.zeros(full_shape, device=device, dtype=torch.float32)
        
        # 2. Extract & Unpack Data
        fmt = w_info['format']
        packed_data = w_info['data']
        row_map = w_info['row_map']
        scale = w_info['scale']
        
        if 'int4' in fmt:
            # Calculate expected elements
            num_elements = len(row_map) * full_shape[1]
            unpacked_data = unpack_int4(packed_data, num_elements)
            # Reshape to [Active_Rows, Cols]
            active_rows_tensor = torch.from_numpy(unpacked_data).reshape(len(row_map), full_shape[1])
        else:
            # INT8
            active_rows_tensor = torch.from_numpy(packed_data).reshape(len(row_map), full_shape[1])
            
        # 3. De-Quantize (Int -> Float) & Inject into correct slots
        reconstructed_tensor[row_map, :] = (active_rows_tensor.float() * scale).to(device)
        
        # 4. Assign to Model
        param.data = reconstructed_tensor
        
    print("✅ Model weights reconstructed and loaded successfully.")
    return model