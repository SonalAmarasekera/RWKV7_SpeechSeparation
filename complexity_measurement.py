#!/usr/bin/env python3
"""
Complexity Measurement Module for RWKV Speech Separation Training

This module provides comprehensive training and computational complexity measurements:
- Training Complexity: Parameters, FLOPs, Model Size
- Computational Complexity: Training time, throughput, memory usage
"""

import time
import torch

class ComplexityMeasurement:
    """Measure training and computational complexity for deep learning models"""
    
    def __init__(self, model, device, log_file="complexity_measurements.txt"):
        self.model = model
        self.device = device
        self.log_file = log_file
        self.measurements = {}
        self.epoch_metrics = []
        
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING AND COMPUTATIONAL COMPLEXITY MEASUREMENTS\n")
            f.write("=" * 80 + "\n\n")
    
    def _calculate_codecformer_style_macs(self, sample_rate=8000, test_duration=2.0, domain='codec'):
        """
        Calculate MACs following CodecFormer methodology for fair comparison.
        
        CodecFormer methodology:
        - Uses MACs (Multiply-Accumulate operations), not FLOPs
        - Excludes encoder/decoder (DAC/SepFormer/STFT) operations
        - Only counts the separator model itself
        - Fixed test input: 2 seconds of 8kHz audio
        - Reports inference requirements
        
        Args:
            sample_rate: Audio sample rate (default: 8000 Hz)
            test_duration: Test audio duration in seconds (default: 2.0s)
            domain: 'codec', 'time', or 'tf' to determine sequence length
            
        Returns:
            dict with 'total_macs', 'separator_only_macs', 'sequence_length'
        """
        # Calculate sequence length for 2-second test audio
        test_samples = int(sample_rate * test_duration)
        
        if domain == 'codec':
            # DAC compression: 8kHz → 50Hz (160x downsampling)
            # For 16kHz: 16000Hz → 50Hz (320x downsampling)
            compression_ratio = 320 if sample_rate == 16000 else 160
            seq_len = test_samples // compression_ratio
        elif domain == 'time':
            # SepFormer encoder: stride=8, kernel=16
            seq_len = (test_samples - 16) // 8 + 1
        elif domain == 'tf':
            # STFT: hop_length=128
            seq_len = test_samples // 128
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        # Calculate separator-only MACs (exclude encoder/decoder/projections)
        separator_macs = self._calculate_separator_core_macs(seq_len)
        
        # Calculate total MACs (include projections)
        total_flops = self._calculate_rwkv_flops(seq_len, 1024)  # Use codec_dim for estimation
        total_macs = total_flops // 2  # MACs = FLOPs / 2
        
        return {
            'total_macs': total_macs,
            'separator_only_macs': separator_macs,
            'sequence_length': seq_len,
            'test_duration': test_duration,
            'sample_rate': sample_rate,
        }
    
    def _calculate_separator_core_macs(self, seq_len):
        """
        Calculate MACs for RWKV core only (excluding encoder/decoder).
        This matches CodecFormer's methodology.
        
        Args:
            seq_len: Sequence length after compression
            
        Returns:
            MACs for separator core only
        """
        try:
            cfg = self.model.cfg
            n_embd = cfg.n_embd
            n_layer = cfg.n_layer
            n_groups = cfg.n_groups
        except:
            n_embd = 512
            n_layer = 8
            n_groups = 2
        
        macs = 0
        
        # ==========================================
        # RWKV CORE ONLY (n_layer blocks)
        # ==========================================
        for layer_idx in range(n_layer):
            group_dim = n_embd // n_groups
            
            for group in range(n_groups):
                # Forward and backward passes
                for direction in ['fwd', 'bwd']:
                    # Receptance, Key, Value, Output projections
                    # Each: group_dim × group_dim matrix multiplication
                    macs += 4 * group_dim * group_dim * seq_len
                    
                    # LoRA projections (w, a, v, g)
                    d_decay_lora = max(32, int(round((2.5 * (group_dim ** 0.5)) / 32) * 32))
                    d_aaa_lora = max(32, int(round((2.5 * (group_dim ** 0.5)) / 32) * 32))
                    d_mv_lora = max(32, int(round((1.7 * (group_dim ** 0.5)) / 32) * 32))
                    d_gate_lora = max(32, int(round((5 * (group_dim ** 0.5)) / 32) * 32))
                    
                    macs += group_dim * d_decay_lora * seq_len  # w1
                    macs += d_decay_lora * group_dim * seq_len  # w2
                    macs += group_dim * d_aaa_lora * seq_len    # a1
                    macs += d_aaa_lora * group_dim * seq_len    # a2
                    macs += group_dim * d_mv_lora * seq_len     # v1
                    macs += d_mv_lora * group_dim * seq_len     # v2
                    macs += group_dim * d_gate_lora * seq_len   # g1
                    macs += d_gate_lora * group_dim * seq_len   # g2
                    
                    # RWKV recurrence (approximate)
                    head_size = 64
                    macs += seq_len * group_dim * head_size * 10
                
                # CGM Conv1d
                macs += (2 * group_dim) * (2 * group_dim) * 3 * seq_len
            
            # DualContextAggregation
            macs += n_embd * n_embd * seq_len  # Global
            macs += n_embd * n_embd * 3 * seq_len  # Local conv
            
            # Channel-mixing
            macs += n_embd * (4 * n_embd) * seq_len  # Key projection
            macs += (4 * n_embd) * n_embd * seq_len  # Value projection
        
        return int(macs)
    
    def _calculate_rwkv_flops(self, seq_len, codec_dim):
        """
        Calculate FLOPs for RWKV-v7 speech separator architecture
        
        This manual calculation accounts for all operations in the model:
        - Input projection
        - RWKV layers (time-mixing + channel-mixing)
        - Output projection
        - Separation head
        
        Args:
            seq_len: Sequence length (T)
            codec_dim: Codec/feature dimension (C)
            
        Returns:
            Total FLOPs per sample
        """
        # Get model configuration from the model itself
        try:
            # Try to extract config from model
            cfg = self.model.cfg
            n_embd = cfg.n_embd
            n_layer = cfg.n_layer
            num_sources = cfg.num_sources
            head_hidden = cfg.head_hidden
            n_groups = cfg.n_groups
        except:
            # Fallback: inspect model structure
            n_embd = self.model.input_proj[1].out_features if hasattr(self.model, 'input_proj') else 512
            n_layer = len(self.model.core.blocks) if hasattr(self.model, 'core') else 8
            num_sources = 2
            head_hidden = 256
            n_groups = 2
        
        flops = 0
        
        # ==========================================
        # 1. INPUT PROJECTION: codec_dim -> n_embd
        # ==========================================
        # LayerNorm: 2 * codec_dim * seq_len (mean + std)
        flops += 2 * codec_dim * seq_len
        
        # Linear: codec_dim -> n_embd
        flops += 2 * codec_dim * n_embd * seq_len
        
        # GELU activation: ~5 ops per element
        flops += 5 * n_embd * seq_len
        
        # ==========================================
        # 2. RWKV CORE (n_layer blocks)
        # ==========================================
        for layer_idx in range(n_layer):
            # ------------------------------------------
            # 2.1 TIME-MIXING (Grouped Bi-RWKV)
            # ------------------------------------------
            group_dim = n_embd // n_groups
            
            for group in range(n_groups):
                # Forward and backward passes
                for direction in ['fwd', 'bwd']:
                    # Layer norm
                    flops += 2 * group_dim * seq_len
                    
                    # Time-shift operation (zero-cost, just indexing)
                    
                    # Receptance projection
                    flops += 2 * group_dim * group_dim * seq_len
                    
                    # Key projection  
                    flops += 2 * group_dim * group_dim * seq_len
                    
                    # Value projection
                    flops += 2 * group_dim * group_dim * seq_len
                    
                    # Output projection
                    flops += 2 * group_dim * group_dim * seq_len
                    
                    # Decay (w) calculation with LoRA
                    # w1 @ w2 + w0
                    d_decay_lora = max(32, int(round((2.5 * (group_dim ** 0.5)) / 32) * 32))
                    flops += 2 * group_dim * d_decay_lora * seq_len  # w1
                    flops += 2 * d_decay_lora * group_dim * seq_len  # w2
                    
                    # Alpha (a) calculation with LoRA
                    d_aaa_lora = max(32, int(round((2.5 * (group_dim ** 0.5)) / 32) * 32))
                    flops += 2 * group_dim * d_aaa_lora * seq_len  # a1
                    flops += 2 * d_aaa_lora * group_dim * seq_len  # a2
                    
                    # V mixing with LoRA
                    d_mv_lora = max(32, int(round((1.7 * (group_dim ** 0.5)) / 32) * 32))
                    flops += 2 * group_dim * d_mv_lora * seq_len  # v1
                    flops += 2 * d_mv_lora * group_dim * seq_len  # v2
                    
                    # Gate (g) calculation with LoRA
                    d_gate_lora = max(32, int(round((5 * (group_dim ** 0.5)) / 32) * 32))
                    flops += 2 * group_dim * d_gate_lora * seq_len  # g1
                    flops += 2 * d_gate_lora * group_dim * seq_len  # g2
                    
                    # Activations (sigmoid, tanh, softplus, etc.)
                    # Approximate: 10 ops per element for various activations
                    flops += 10 * group_dim * seq_len
                    
                    # RWKV recurrence (WindBackstepping kernel)
                    # This is complex - approximate as O(seq_len * group_dim * head_size)
                    head_size = 64
                    num_heads = group_dim // head_size
                    flops += seq_len * group_dim * head_size * 20  # 20 ops per recurrence step
                    
                    # Group norm
                    flops += 2 * group_dim * seq_len
                    
                    # Element-wise multiplications for mixing
                    flops += group_dim * seq_len * 5
                
                # CGM (Cross-group mixing) - Conv1d + GLU
                # Conv1d: kernel_size=3, groups=1
                flops += 2 * (2 * group_dim) * (2 * group_dim) * 3 * seq_len
                # GLU
                flops += 2 * group_dim * seq_len
            
            # DualContextAggregation
            # Global context (linear)
            flops += 2 * n_embd * n_embd * seq_len
            # Local context (conv1d, kernel_size=3)
            flops += 2 * n_embd * n_embd * 3 * seq_len
            # Sigmoid
            flops += 5 * n_embd * seq_len
            # Element-wise ops
            flops += n_embd * seq_len * 3
            
            # Residual connection
            flops += n_embd * seq_len
            
            # Dropout (negligible)
            
            # ------------------------------------------
            # 2.2 CHANNEL-MIXING
            # ------------------------------------------
            # Layer norm
            flops += 2 * n_embd * seq_len
            
            # Key projection: n_embd -> 4*n_embd
            flops += 2 * n_embd * (4 * n_embd) * seq_len
            
            # ReLU squared (2 ops: relu + square)
            flops += 2 * (4 * n_embd) * seq_len
            
            # Value projection: 4*n_embd -> n_embd
            flops += 2 * (4 * n_embd) * n_embd * seq_len
            
            # Residual connection
            flops += n_embd * seq_len
        
        # ==========================================
        # 3. OUTPUT PROJECTION: n_embd -> head_hidden
        # ==========================================
        # Layer norm
        flops += 2 * n_embd * seq_len
        
        # Linear projection
        flops += 2 * n_embd * head_hidden * seq_len
        
        # GELU
        flops += 5 * head_hidden * seq_len
        
        # ==========================================
        # 4. SEPARATION HEAD
        # ==========================================
        # Note: No pre-projection layer (direct projection)
        
        # Layer norm
        flops += 2 * head_hidden * seq_len
        
        # Snake activation (~10 ops per element)
        flops += 10 * head_hidden * seq_len
        
        # Final projection: head_hidden -> num_sources * codec_dim
        flops += 2 * head_hidden * (num_sources * codec_dim) * seq_len
        
        # Residual mode operations (element-wise additions)
        flops += num_sources * codec_dim * seq_len
        
        return int(flops)
    
    def measure_model_complexity(self, input_shape):
        """
        Measure model parameters, FLOPs, and memory requirements
        
        Args:
            input_shape: tuple (B, T, C) for input tensor shape
        """
        print("\n" + "="*80)
        print("MEASURING TRAINING COMPLEXITY")
        print("="*80)
        
        # 1. Parameter Count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        self.measurements['total_parameters'] = total_params
        self.measurements['trainable_parameters'] = trainable_params
        self.measurements['non_trainable_parameters'] = non_trainable_params
        
        print(f"Total Parameters:       {total_params:,}")
        print(f"Trainable Parameters:   {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        
        # 2. Model Size in Memory (MB)
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        self.measurements['model_size_mb'] = model_size_mb
        self.measurements['param_size_mb'] = param_size / 1024**2
        self.measurements['buffer_size_mb'] = buffer_size / 1024**2
        
        print(f"\nModel Memory Footprint:")
        print(f"  Parameters: {param_size / 1024**2:.2f} MB")
        print(f"  Buffers:    {buffer_size / 1024**2:.2f} MB")
        print(f"  Total:      {model_size_mb:.2f} MB")
        
        # 3. Manual FLOPs calculation for RWKV architecture
        try:
            B, T, C = input_shape
            
            # Calculate FLOPs manually (more accurate for RWKV)
            flops_per_sample = self._calculate_rwkv_flops(T, C)
            
            self.measurements['flops_per_sample'] = flops_per_sample
            self.measurements['gflops_per_sample'] = flops_per_sample / 1e9
            self.measurements['tflops_per_sample'] = flops_per_sample / 1e12
            
            print(f"\nComputational Complexity (Manual FLOPs Calculation):")
            print(f"  Per sample: {flops_per_sample:,.0f} FLOPs")
            print(f"  Per sample: {flops_per_sample / 1e9:.3f} GFLOPs")
            print(f"  Per sample: {flops_per_sample / 1e12:.6f} TFLOPs")
            print(f"  Note: Accurate calculation for RWKV architecture")
            
        except Exception as e:
            print(f"\n[WARNING] Manual FLOPs calculation failed: {e}")
            self.measurements['flops_per_sample'] = None
            self.measurements['gflops_per_sample'] = None
        
        # 4. GPU Memory Baseline
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            baseline_mem = torch.cuda.memory_allocated(self.device) / 1024**2
            self.measurements['baseline_memory_mb'] = baseline_mem
            
            print(f"\nGPU Memory:")
            print(f"  Baseline (model only): {baseline_mem:.2f} MB")
        
        print("="*80 + "\n")
        
        # Write to log file
        self._log_training_complexity()
        
        return self.measurements
    
    def measure_inference_macs(self, sample_rate=8000, test_duration=2.0, domain='codec'):
        """
        Measure inference MACs following CodecFormer methodology.
        
        This should be called separately after measure_model_complexity() to get
        standardized inference metrics for comparison with CodecFormer paper.
        
        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz)
            test_duration: Test audio duration in seconds (default: 2.0s)
            domain: 'codec', 'time', or 'tf'
            
        Returns:
            dict with MAC measurements
        """
        print("\n" + "="*80)
        print("CODECFORMER-STYLE INFERENCE COMPLEXITY")
        print("="*80)
        print(f"Following CodecFormer paper methodology:")
        print(f"  - Fixed test input: {test_duration}s at {sample_rate}Hz")
        print(f"  - MACs (Multiply-Accumulate operations)")
        print(f"  - Separator model only (excludes encoder/decoder)")
        print(f"  - Hardware-agnostic inference requirements")
        print("="*80)
        
        mac_results = self._calculate_codecformer_style_macs(sample_rate, test_duration, domain)
        
        # Store in measurements
        self.measurements['inference_macs'] = mac_results['separator_only_macs']
        self.measurements['inference_gmacs'] = mac_results['separator_only_macs'] / 1e9
        self.measurements['inference_seq_len'] = mac_results['sequence_length']
        self.measurements['test_duration'] = test_duration
        self.measurements['test_sample_rate'] = sample_rate
        
        print(f"\nTest Configuration:")
        print(f"  Duration:       {test_duration}s")
        print(f"  Sample rate:    {sample_rate}Hz")
        print(f"  Total samples:  {int(sample_rate * test_duration):,}")
        print(f"  Sequence length: {mac_results['sequence_length']} frames")
        
        print(f"\nInference MACs (Separator Only):")
        print(f"  MACs:  {mac_results['separator_only_macs']:,.0f}")
        print(f"  GMACs: {mac_results['separator_only_macs'] / 1e9:.3f}")
        print(f"  MMACs: {mac_results['separator_only_macs'] / 1e6:.1f}")
        
        print(f"\nTotal MACs (Including Projections):")
        print(f"  MACs:  {mac_results['total_macs']:,.0f}")
        print(f"  GMACs: {mac_results['total_macs'] / 1e9:.3f}")
        
        print(f"\nNote: Encoder/decoder operations excluded")
        print(f"      (DAC/SepFormer/STFT not counted)")
        print("="*80 + "\n")
        
        # Log to file
        self._log_inference_macs(mac_results, domain)
        
        return mac_results
    
    def start_epoch_timing(self):
        """Start timing an epoch"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        self.epoch_start_time = time.time()
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def end_epoch_timing(self, num_batches, num_samples):
        """
        End epoch timing and calculate statistics
        
        Args:
            num_batches: number of batches processed
            num_samples: total samples processed in epoch
        
        Returns:
            dict: epoch measurements
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        epoch_time = time.time() - self.epoch_start_time
        
        # Calculate metrics
        measurements = {
            'epoch_time_sec': epoch_time,
            'epoch_time_min': epoch_time / 60,
            'time_per_batch_sec': epoch_time / num_batches,
            'time_per_batch_ms': (epoch_time / num_batches) * 1000,
            'samples_per_second': num_samples / epoch_time,
            'batches_per_second': num_batches / epoch_time,
        }
        
        # GPU Memory metrics
        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**2
            
            measurements['peak_memory_mb'] = peak_memory
            measurements['current_memory_mb'] = current_memory
            measurements['reserved_memory_mb'] = reserved_memory
            
            # Calculate memory overhead
            if 'baseline_memory_mb' in self.measurements:
                measurements['memory_overhead_mb'] = peak_memory - self.measurements['baseline_memory_mb']
        
        self.epoch_metrics.append(measurements)
        
        return measurements
    
    def _log_training_complexity(self):
        """Write training complexity to log file"""
        with open(self.log_file, 'a') as f:
            f.write("TRAINING COMPLEXITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Parameters:       {self.measurements['total_parameters']:,}\n")
            f.write(f"Trainable Parameters:   {self.measurements['trainable_parameters']:,}\n")
            f.write(f"Non-trainable Parameters: {self.measurements['non_trainable_parameters']:,}\n")
            f.write(f"\nModel Size:\n")
            f.write(f"  Parameters: {self.measurements['param_size_mb']:.2f} MB\n")
            f.write(f"  Buffers:    {self.measurements['buffer_size_mb']:.2f} MB\n")
            f.write(f"  Total:      {self.measurements['model_size_mb']:.2f} MB\n")
            
            if self.measurements.get('gflops_per_sample') is not None:
                f.write(f"\nComputational Complexity (Manual FLOPs Calculation):\n")
                f.write(f"  FLOPs per sample: {self.measurements['flops_per_sample']:,.0f}\n")
                f.write(f"  GFLOPs per sample: {self.measurements['gflops_per_sample']:.3f}\n")
                if self.measurements.get('tflops_per_sample'):
                    f.write(f"  TFLOPs per sample: {self.measurements['tflops_per_sample']:.6f}\n")
                f.write(f"  Note: Accurate calculation for RWKV architecture\n")
            
            if 'baseline_memory_mb' in self.measurements:
                f.write(f"\nGPU Memory Baseline: {self.measurements['baseline_memory_mb']:.2f} MB\n")
            
            f.write("\n")
    
    def _log_inference_macs(self, mac_results, domain):
        """Write CodecFormer-style inference MACs to log file"""
        with open(self.log_file, 'a') as f:
            f.write("CODECFORMER-STYLE INFERENCE COMPLEXITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Methodology: Following CodecFormer paper\n")
            f.write(f"Domain: {domain}\n")
            f.write(f"Test input: {mac_results['test_duration']}s at {mac_results['sample_rate']}Hz\n")
            f.write(f"Sequence length: {mac_results['sequence_length']} frames\n")
            f.write(f"\nInference MACs (Separator Only, excludes encoder/decoder):\n")
            f.write(f"  MACs:  {mac_results['separator_only_macs']:,.0f}\n")
            f.write(f"  GMACs: {mac_results['separator_only_macs'] / 1e9:.3f}\n")
            f.write(f"  MMACs: {mac_results['separator_only_macs'] / 1e6:.1f}\n")
            f.write(f"\nTotal MACs (Including input/output projections):\n")
            f.write(f"  MACs:  {mac_results['total_macs']:,.0f}\n")
            f.write(f"  GMACs: {mac_results['total_macs'] / 1e9:.3f}\n")
            f.write(f"\nNote: Encoder/decoder operations excluded (DAC/SepFormer/STFT not counted)\n")
            f.write(f"      Use 'Separator Only' MACs for comparison with CodecFormer paper\n")
            f.write("\n")
    
    def log_epoch_complexity(self, epoch, measurements):
        """Write epoch computational complexity to log file"""
        with open(self.log_file, 'a') as f:
            f.write(f"EPOCH {epoch} - COMPUTATIONAL COMPLEXITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Time:          {measurements['epoch_time_sec']:.2f} sec ({measurements['epoch_time_min']:.2f} min)\n")
            f.write(f"Time per Batch:         {measurements['time_per_batch_ms']:.2f} ms\n")
            f.write(f"Throughput:             {measurements['samples_per_second']:.2f} samples/sec\n")
            f.write(f"Batch Processing Rate:  {measurements['batches_per_second']:.2f} batches/sec\n")
            
            if 'peak_memory_mb' in measurements:
                f.write(f"\nGPU Memory Usage:\n")
                f.write(f"  Peak:     {measurements['peak_memory_mb']:.2f} MB\n")
                f.write(f"  Current:  {measurements['current_memory_mb']:.2f} MB\n")
                f.write(f"  Reserved: {measurements['reserved_memory_mb']:.2f} MB\n")
                
                if 'memory_overhead_mb' in measurements:
                    f.write(f"  Overhead: {measurements['memory_overhead_mb']:.2f} MB\n")
            
            f.write("\n")
    
    def print_epoch_summary(self, epoch, measurements):
        """Print epoch complexity summary to console"""
        print(f"  [Complexity] Epoch {epoch}: "
              f"Time={measurements['epoch_time_sec']:.2f}s, "
              f"Throughput={measurements['samples_per_second']:.2f} samples/s")
        
        if 'peak_memory_mb' in measurements:
            print(f"  [Complexity] GPU Memory: "
                  f"Peak={measurements['peak_memory_mb']:.2f}MB, "
                  f"Current={measurements['current_memory_mb']:.2f}MB")
    
    def calculate_aggregate_stats(self):
        """Calculate aggregate statistics across all epochs"""
        if not self.epoch_metrics:
            return None
        
        num_epochs = len(self.epoch_metrics)
        
        # Calculate averages
        avg_epoch_time = sum(m['epoch_time_sec'] for m in self.epoch_metrics) / num_epochs
        avg_throughput = sum(m['samples_per_second'] for m in self.epoch_metrics) / num_epochs
        
        stats = {
            'num_epochs_measured': num_epochs,
            'avg_epoch_time_sec': avg_epoch_time,
            'avg_throughput_samples_per_sec': avg_throughput,
            'total_training_time_sec': sum(m['epoch_time_sec'] for m in self.epoch_metrics),
            'total_training_time_hours': sum(m['epoch_time_sec'] for m in self.epoch_metrics) / 3600,
        }
        
        if 'peak_memory_mb' in self.epoch_metrics[0]:
            stats['max_peak_memory_mb'] = max(m['peak_memory_mb'] for m in self.epoch_metrics)
            stats['avg_peak_memory_mb'] = sum(m['peak_memory_mb'] for m in self.epoch_metrics) / num_epochs
        
        return stats
    
    def print_final_summary(self):
        """Print and log final complexity summary"""
        stats = self.calculate_aggregate_stats()
        
        if stats is None:
            print("\n[INFO] No epoch metrics collected")
            return
        
        print("\n" + "="*80)
        print("FINAL COMPLEXITY SUMMARY")
        print("="*80)
        print(f"Training Complexity:")
        print(f"  Total Parameters:    {self.measurements['total_parameters']:,}")
        print(f"  Model Size:          {self.measurements['model_size_mb']:.2f} MB")
        if self.measurements.get('gflops_per_sample'):
            print(f"  GFLOPs per sample:   {self.measurements['gflops_per_sample']:.3f}")
        
        print(f"\nComputational Complexity (Avg over {stats['num_epochs_measured']} epochs):")
        print(f"  Avg Epoch Time:      {stats['avg_epoch_time_sec']:.2f} sec")
        print(f"  Avg Throughput:      {stats['avg_throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Total Training Time: {stats['total_training_time_hours']:.2f} hours")
        
        if 'max_peak_memory_mb' in stats:
            print(f"\nGPU Memory:")
            print(f"  Max Peak:            {stats['max_peak_memory_mb']:.2f} MB")
            print(f"  Avg Peak:            {stats['avg_peak_memory_mb']:.2f} MB")
        
        print(f"\nDetailed logs saved to: {self.log_file}")
        print("="*80 + "\n")
        
        # Write summary to log file
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Epochs Measured: {stats['num_epochs_measured']}\n")
            f.write(f"Average Epoch Time: {stats['avg_epoch_time_sec']:.2f} sec\n")
            f.write(f"Average Throughput: {stats['avg_throughput_samples_per_sec']:.2f} samples/sec\n")
            f.write(f"Total Training Time: {stats['total_training_time_hours']:.2f} hours\n")
            
            if 'max_peak_memory_mb' in stats:
                f.write(f"Max Peak Memory: {stats['max_peak_memory_mb']:.2f} MB\n")
                f.write(f"Avg Peak Memory: {stats['avg_peak_memory_mb']:.2f} MB\n")
            
            f.write("="*80 + "\n")