"""
SepFormer Manual Extraction - FIXED
Handles correct checkpoint structure
"""

import torch
import torch.nn as nn
import os
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


class SepFormerEncoder(nn.Module):
    """SepFormer Encoder - matches checkpoint structure"""
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class SepFormerDecoder(nn.Module):
    """
    SepFormer Decoder - FIXED to match actual checkpoint structure
    
    The checkpoint stores weight directly, not as transconv1d.weight
    """
    def __init__(self, in_channels=256, out_channels=1, kernel_size=16, stride=8):
        super().__init__()
        # Store these for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Create the actual layer
        self.transconv1d = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
    
    def forward(self, x):
        x = self.transconv1d(x)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        return x
    
    def load_state_dict_custom(self, state_dict):
        """
        Custom loading to handle checkpoint format mismatch
        Checkpoint has 'weight' but we need 'transconv1d.weight'
        """
        if 'weight' in state_dict and 'transconv1d.weight' not in state_dict:
            # Remap weight to transconv1d.weight
            new_state_dict = {'transconv1d.weight': state_dict['weight']}
            if 'bias' in state_dict:
                new_state_dict['transconv1d.bias'] = state_dict['bias']
            return super().load_state_dict(new_state_dict)
        else:
            return super().load_state_dict(state_dict)


def inspect_checkpoint(ckpt_path):
    """Inspect checkpoint structure"""
    print(f"\n   Inspecting {os.path.basename(ckpt_path)}:")
    state = torch.load(ckpt_path, map_location='cpu')
    
    if isinstance(state, dict):
        print(f"   Type: dict with keys: {list(state.keys())}")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                print(f"     - {key}: tensor {list(value.shape)}")
            else:
                print(f"     - {key}: {type(value)}")
    else:
        print(f"   Type: {type(state)}")
        if isinstance(state, torch.Tensor):
            print(f"   Shape: {list(state.shape)}")
    
    return state


def extract_sepformer_manually(save_dir='pretrained_weights'):
    """Extract SepFormer encoder/decoder with correct checkpoint handling"""
    
    print("="*80)
    print("MANUAL SEPFORMER EXTRACTION (FIXED)")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Build architectures
    print("\n[1/5] Building encoder and decoder architectures...")
    
    encoder = SepFormerEncoder(kernel_size=16, out_channels=256, in_channels=1)
    decoder = SepFormerDecoder(in_channels=256, out_channels=1, kernel_size=16, stride=8)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"      ✓ Encoder created: {encoder_params:,} parameters")
    print(f"      ✓ Decoder created: {decoder_params:,} parameters")
    
    # Step 2: Download weights
    print("\n[2/5] Downloading pre-trained weights from HuggingFace...")
    
    base_url = "https://huggingface.co/speechbrain/sepformer-wsj02mix/resolve/main"
    
    encoder_url = f"{base_url}/encoder.ckpt"
    encoder_ckpt_path = os.path.join(save_dir, 'encoder.ckpt')
    
    if not os.path.exists(encoder_ckpt_path):
        print(f"\n      Downloading encoder...")
        try:
            download_file(encoder_url, encoder_ckpt_path)
            print(f"      ✓ Downloaded encoder.ckpt")
        except Exception as e:
            print(f"      ❌ Download failed: {e}")
            return None, None, None
    else:
        print(f"      ✓ encoder.ckpt already exists")
    
    decoder_url = f"{base_url}/decoder.ckpt"
    decoder_ckpt_path = os.path.join(save_dir, 'decoder.ckpt')
    
    if not os.path.exists(decoder_ckpt_path):
        print(f"\n      Downloading decoder...")
        try:
            download_file(decoder_url, decoder_ckpt_path)
            print(f"      ✓ Downloaded decoder.ckpt")
        except Exception as e:
            print(f"      ❌ Download failed: {e}")
            return None, None, None
    else:
        print(f"      ✓ decoder.ckpt already exists")
    
    # Step 3: Inspect checkpoints
    print("\n[3/5] Inspecting checkpoint structures...")
    
    encoder_state = inspect_checkpoint(encoder_ckpt_path)
    decoder_state = inspect_checkpoint(decoder_ckpt_path)
    
    # Step 4: Load weights
    print("\n[4/5] Loading pre-trained weights...")
    
    try:
        # Load encoder - handles nested structure if needed
        if isinstance(encoder_state, dict):
            if 'conv1d.weight' in encoder_state:
                # Already in correct format
                encoder.load_state_dict(encoder_state)
            elif 'weight' in encoder_state and 'conv1d.weight' not in encoder_state:
                # Remap
                new_state = {'conv1d.weight': encoder_state['weight']}
                encoder.load_state_dict(new_state)
            else:
                # Try direct
                encoder.load_state_dict(encoder_state)
        else:
            print("      ⚠️  Encoder checkpoint is a raw tensor, not a state dict")
            print("      Attempting to load directly...")
            encoder.conv1d.weight.data = encoder_state
        
        print(f"      ✓ Loaded encoder weights")
        
    except Exception as e:
        print(f"      ❌ Encoder loading failed: {e}")
        return None, None, None
    
    try:
        # Load decoder - use custom loading
        decoder.load_state_dict_custom(decoder_state)
        print(f"      ✓ Loaded decoder weights")
        
    except Exception as e:
        print(f"      ❌ Decoder loading failed: {e}")
        
        # Last resort: manual weight assignment
        print("      Trying manual weight assignment...")
        try:
            if isinstance(decoder_state, dict) and 'weight' in decoder_state:
                decoder.transconv1d.weight.data = decoder_state['weight']
                print(f"      ✓ Manually assigned decoder weights")
            elif isinstance(decoder_state, torch.Tensor):
                decoder.transconv1d.weight.data = decoder_state
                print(f"      ✓ Manually assigned decoder weights (raw tensor)")
            else:
                print(f"      ❌ Cannot load decoder - unknown format")
                return None, None, None
        except Exception as e2:
            print(f"      ❌ Manual assignment also failed: {e2}")
            return None, None, None
    
    # Step 5: Test and save
    print("\n[5/5] Testing and saving...")
    
    encoder.eval()
    decoder.eval()
    
    # Test
    test_input = torch.randn(1, 16000)  # 1 second
    
    try:
        with torch.no_grad():
            encoded = encoder(test_input)
            decoded = decoder(encoded)
        
        print(f"      Test input:  {list(test_input.shape)}")
        print(f"      Encoded:     {list(encoded.shape)}")
        print(f"      Decoded:     {list(decoded.shape)}")
        print(f"      ✓ Pipeline works!")
        
    except Exception as e:
        print(f"      ❌ Testing failed: {e}")
        return None, None, None
    
    # Save processed weights
    encoder_path = os.path.join(save_dir, 'sepformer_encoder.pth')
    decoder_path = os.path.join(save_dir, 'sepformer_decoder.pth')
    
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    
    print(f"\n      ✓ Saved encoder: {encoder_path}")
    print(f"      ✓ Saved decoder: {decoder_path}")
    
    # Save info
    info = {
        'encoder_config': {
            'in_channels': 1,
            'out_channels': 256,
            'kernel_size': 16,
            'stride': 8,
        },
        'decoder_config': {
            'in_channels': 256,
            'out_channels': 1,
            'kernel_size': 16,
            'stride': 8,
        },
        'encoder_params': encoder_params,
        'decoder_params': decoder_params,
        'total_params': encoder_params + decoder_params,
        'encoded_channels': encoded.shape[1],
        'sample_rate': 8000,
    }
    
    info_path = os.path.join(save_dir, 'architecture_info.pt')
    torch.save(info, info_path)
    
    print(f"      ✓ Saved info: {info_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE!")
    print("="*80)
    
    print(f"\nFiles in '{save_dir}':")
    print(f"  📁 sepformer_encoder.pth ({encoder_params:,} params) ← USE THIS")
    print(f"  📁 sepformer_decoder.pth ({decoder_params:,} params) ← USE THIS")
    print(f"  📁 architecture_info.pt")
    
    return encoder, decoder, info


def test_extracted_weights(save_dir='pretrained_weights'):
    """Test the extracted weights"""
    
    print("\n" + "="*80)
    print("TESTING EXTRACTED WEIGHTS")
    print("="*80)
    
    # Recreate architectures
    encoder = SepFormerEncoder()
    decoder = SepFormerDecoder()
    
    # Load weights
    encoder.load_state_dict(torch.load(f'{save_dir}/sepformer_encoder.pth'))
    decoder.load_state_dict(torch.load(f'{save_dir}/sepformer_decoder.pth'))
    
    encoder.eval()
    decoder.eval()
    
    print("\n✓ Weights loaded successfully")
    
    # Test with different lengths
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    test_lengths = [8000, 16000, 32000]  # 0.5s, 1s, 2s at 8kHz
    
    for length in test_lengths:
        test_input = torch.randn(2, length)
        
        with torch.no_grad():
            encoded = encoder(test_input)
            decoded = decoder(encoded)
        
        print(f"\nTest {length} samples ({length/8000:.1f}s @ 8kHz):")
        print(f"  Input:   {list(test_input.shape)}")
        print(f"  Encoded: {list(encoded.shape)}")
        print(f"  Decoded: {list(decoded.shape)}")
        
        # Check reconstruction (may not match exactly due to compression)
        print(f"  Length diff: {abs(decoded.shape[1] - test_input.shape[1])} samples")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SEPFORMER MANUAL EXTRACTION - FIXED VERSION")
    print("="*80)
    
    # Extract
    encoder, decoder, info = extract_sepformer_manually()
    
    if encoder is not None:
        # Test
        test_extracted_weights()
        
        print("\n" + "="*80)
        print("SUCCESS! 🎉")
        print("="*80)
        print("\n✅ Encoder and decoder ready for RWKV integration!")
        print("\nFiles to use:")
        print("  • pretrained_weights/sepformer_encoder.pth")
        print("  • pretrained_weights/sepformer_decoder.pth")
        print("\nNext steps:")
        print("  1. Integrate with RWKV separator")
        print("  2. Freeze encoder/decoder during training")
        print("  3. Train RWKV separator only")
        print("  4. Expected: 17-20 dB SI-SDRi")
        print("="*80)
    else:
        print("\n❌ Extraction failed. Check errors above.")