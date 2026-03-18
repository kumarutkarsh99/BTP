"""
Advanced Mixed-Precision Search
- Greedy search
- Simulated Annealing
- Sensitivity caching
- NAS-based search (RL)
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SENSITIVITY CACHE
# ============================================================================

class SensitivityCache:
    """Cache sensitivity scores to avoid re-evaluation."""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, layer_name, precision):
        """Get cached sensitivity."""
        key = f"{layer_name}_{precision}"
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def store(self, layer_name, precision, sensitivity):
        """Store sensitivity score."""
        key = f"{layer_name}_{precision}"
        self.cache[key] = sensitivity
    
    def stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({hit_rate:.1%})"


# ============================================================================
# GREEDY SEARCH
# ============================================================================

def run_greedy_search(model, val_loader, device, sensitivity_threshold=0.015):
    """
    Greedy layer-wise search for mixed precision.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        sensitivity_threshold: Max accuracy drop allowed
    
    Returns:
        tuple: (final_accuracy, precision_profile)
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    logger.info("Greedy mixed-precision search")
    
    # Get all quantizable layers
    layer_names = [n for n, p in model.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    
    # Save original weights
    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # Start with all INT8
    profile = {n: 'INT8' for n in layer_names}
    apply_quantization_to_model(model, profile)
    baseline_acc = get_accuracy(model, val_loader, device)
    acc_limit = baseline_acc - sensitivity_threshold
    
    logger.info(f"Baseline INT8: {baseline_acc:.4f}, Limit: {acc_limit:.4f}")
    
    # Calculate sensitivity for each layer
    sensitivities = []
    
    for i, layer in enumerate(layer_names):
        # Restore weights
        for n, p in model.named_parameters():
            p.data.copy_(original_weights[n])
        
        # Test INT4 for this layer
        test_profile = profile.copy()
        test_profile[layer] = 'INT4'
        
        apply_quantization_to_model(model, test_profile)
        acc = get_accuracy(model, val_loader, device)
        
        sensitivity = baseline_acc - acc
        sensitivities.append((layer, sensitivity))
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Analyzed {i+1}/{len(layer_names)} layers")
    
    # Sort by sensitivity (least sensitive first)
    sensitivities.sort(key=lambda x: x[1])
    
    # Greedy selection
    final_profile = profile.copy()
    
    for layer, sensitivity in sensitivities:
        # Restore weights
        for n, p in model.named_parameters():
            p.data.copy_(original_weights[n])
        
        # Try INT4
        final_profile[layer] = 'INT4'
        apply_quantization_to_model(model, final_profile)
        acc = get_accuracy(model, val_loader, device)
        
        if acc < acc_limit:
            # Revert to INT8
            final_profile[layer] = 'INT8'
    
    # Final evaluation
    for n, p in model.named_parameters():
        p.data.copy_(original_weights[n])
    
    apply_quantization_to_model(model, final_profile)
    final_acc = get_accuracy(model, val_loader, device)
    
    num_int4 = sum(1 for v in final_profile.values() if v == 'INT4')
    logger.info(f"✅ Greedy: {num_int4}/{len(layer_names)} INT4, Acc={final_acc:.4f}")
    
    return final_acc, final_profile


# ============================================================================
# SIMULATED ANNEALING SEARCH
# ============================================================================

def run_simulated_annealing(model, val_loader, device,
                            max_iterations=100, initial_temp=1.0,
                            cooling_rate=0.95, sensitivity_threshold=0.015,
                            use_cache=True):
    """
    Simulated Annealing for better mixed-precision search.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        max_iterations: Number of SA iterations
        initial_temp: Starting temperature
        cooling_rate: Temperature decay rate
        sensitivity_threshold: Max accuracy drop
        use_cache: Use sensitivity caching
    
    Returns:
        tuple: (final_accuracy, precision_profile)
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    logger.info(f"Simulated Annealing search ({max_iterations} iterations)")
    
    # Get layers
    layer_names = [n for n, p in model.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    
    # Save original weights
    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # Initialize cache
    cache = SensitivityCache() if use_cache else None
    
    # Start with all INT8
    current_profile = {n: 'INT8' for n in layer_names}
    apply_quantization_to_model(model, current_profile)
    baseline_acc = get_accuracy(model, val_loader, device)
    acc_limit = baseline_acc - sensitivity_threshold
    
    # Track best solution
    best_profile = current_profile.copy()
    best_score = calculate_score(current_profile, baseline_acc, baseline_acc)
    
    temperature = initial_temp
    current_score = best_score
    
    logger.info(f"Baseline: INT8={baseline_acc:.4f}, Limit={acc_limit:.4f}")
    
    # SA loop
    for iteration in range(max_iterations):
        # Restore weights
        for n, p in model.named_parameters():
            p.data.copy_(original_weights[n])
        
        # Generate neighbor (flip one layer's precision)
        neighbor_profile = current_profile.copy()
        random_layer = np.random.choice(layer_names)
        
        if neighbor_profile[random_layer] == 'INT8':
            neighbor_profile[random_layer] = 'INT4'
        else:
            neighbor_profile[random_layer] = 'INT8'
        
        # Check cache
        if cache:
            cached_acc = cache.get(random_layer, neighbor_profile[random_layer])
            if cached_acc is not None:
                neighbor_acc = cached_acc
            else:
                apply_quantization_to_model(model, neighbor_profile)
                neighbor_acc = get_accuracy(model, val_loader, device)
                cache.store(random_layer, neighbor_profile[random_layer], neighbor_acc)
        else:
            apply_quantization_to_model(model, neighbor_profile)
            neighbor_acc = get_accuracy(model, val_loader, device)
        
        # Accept/reject
        if neighbor_acc >= acc_limit:
            neighbor_score = calculate_score(neighbor_profile, neighbor_acc, baseline_acc)
            delta = neighbor_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_profile = neighbor_profile
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_profile = current_profile.copy()
                    best_score = current_score
                    num_int4 = sum(1 for v in best_profile.values() if v == 'INT4')
                    logger.info(f"  Iter {iteration+1}: New best - "
                              f"INT4={num_int4}/{len(layer_names)}, "
                              f"Score={best_score:.4f}")
        
        # Cool down
        temperature *= cooling_rate
        
        if (iteration + 1) % 20 == 0:
            logger.info(f"  Iteration {iteration+1}/{max_iterations}, Temp={temperature:.3f}")
    
    # Apply best profile
    for n, p in model.named_parameters():
        p.data.copy_(original_weights[n])
    
    apply_quantization_to_model(model, best_profile)
    final_acc = get_accuracy(model, val_loader, device)
    
    num_int4 = sum(1 for v in best_profile.values() if v == 'INT4')
    
    if cache:
        logger.info(f"  {cache.stats()}")
    
    logger.info(f"✅ SA: {num_int4}/{len(layer_names)} INT4 ({num_int4/len(layer_names)*100:.1f}%), Acc={final_acc:.4f}")
    
    return final_acc, best_profile


def calculate_score(profile, accuracy, baseline_acc):
    """Calculate score for SA (higher = better)."""
    int4_ratio = sum(1 for v in profile.values() if v == 'INT4') / len(profile)
    acc_score = accuracy / baseline_acc
    return acc_score + 0.3 * int4_ratio  # Reward INT4 usage


# ============================================================================
# NAS-BASED SEARCH (RL Agent)
# ============================================================================

class PrecisionSearchAgent(nn.Module):
    """RL agent for precision assignment."""
    
    def __init__(self, num_layers, num_precisions=3, embedding_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.num_precisions = num_precisions
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(10, embedding_dim),  # 10 features per layer
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_precisions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, layer_features):
        """
        Args:
            layer_features: [num_layers, 10]
        
        Returns:
            precision_probs: [num_layers, num_precisions]
        """
        return self.policy(layer_features)
    
    def sample_profile(self, layer_features, layer_names):
        """Sample precision profile."""
        probs = self.forward(layer_features)
        
        profile = {}
        log_probs = []
        
        precisions = ['INT4', 'INT8', 'INT16']
        
        for i, layer_name in enumerate(layer_names):
            precision_idx = torch.multinomial(probs[i], 1).item()
            profile[layer_name] = precisions[precision_idx]
            log_probs.append(torch.log(probs[i, precision_idx]))
        
        return profile, torch.stack(log_probs)


def run_nas_search(model, train_loader, val_loader, device,
                  episodes=50, lr=1e-3):
    """
    NAS-based search using RL.
    
    Args:
        model: Model to quantize
        train_loader: Training data (for calibration)
        val_loader: Validation data
        device: Device
        episodes: Number of training episodes
        lr: Learning rate for agent
    
    Returns:
        tuple: (final_accuracy, precision_profile)
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    from utils.utils import extract_layer_features
    
    logger.info(f"NAS-based search ({episodes} episodes)")
    
    # Get layers
    layer_names = [n for n, p in model.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    
    # Extract layer features
    layer_features = extract_layer_features(model, layer_names).to(device)
    
    # Save original weights
    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # Create agent
    agent = PrecisionSearchAgent(num_layers=len(layer_names)).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    best_profile = None
    best_reward = -float('inf')
    
    # Training loop
    for episode in range(episodes):
        # Sample profile
        profile, log_probs = agent.sample_profile(layer_features, layer_names)
        
        # Restore weights
        for n, p in model.named_parameters():
            p.data.copy_(original_weights[n])
        
        # Apply quantization
        apply_quantization_to_model(model, profile)
        
        # Evaluate
        acc = get_accuracy(model, val_loader, device)
        
        # Calculate reward (accuracy - size penalty)
        int4_ratio = sum(1 for v in profile.values() if v == 'INT4') / len(profile)
        reward = acc - 0.001 * (1 - int4_ratio)  # Encourage INT4
        
        # Policy gradient update
        loss = -(log_probs.sum() * reward)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track best
        if reward > best_reward:
            best_reward = reward
            best_profile = profile.copy()
            logger.info(f"  Episode {episode+1}: Reward={reward:.4f}, Acc={acc:.4f}")
    
    # Apply best profile
    for n, p in model.named_parameters():
        p.data.copy_(original_weights[n])
    
    apply_quantization_to_model(model, best_profile)
    final_acc = get_accuracy(model, val_loader, device)
    
    num_int4 = sum(1 for v in best_profile.values() if v == 'INT4')
    logger.info(f"✅ NAS: {num_int4}/{len(layer_names)} INT4, Acc={final_acc:.4f}")
    
    return final_acc, best_profile


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def run_hybrid_search(model, val_loader, device, method='simulated_annealing',
                     **kwargs):
    """
    Unified interface for all search methods.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        method: 'greedy', 'simulated_annealing', 'nas'
        **kwargs: Method-specific arguments
    
    Returns:
        tuple: (final_accuracy, precision_profile)
    """
    if method == 'greedy':
        return run_greedy_search(model, val_loader, device, **kwargs)
    elif method == 'simulated_annealing':
        return run_simulated_annealing(model, val_loader, device, **kwargs)
    elif method == 'nas':
        train_loader = kwargs.pop('train_loader', None)
        return run_nas_search(model, train_loader, val_loader, device, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
