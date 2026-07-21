"""
Quick test script to verify attention heatmap functionality
"""
import os
import sys
import numpy as np
import torch

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.image_module.vit_model import FireImageDetector

def test_attention_heatmap():
    """Test the attention heatmap generation with a real image."""
    print("=" * 60)
    print("Testing Attention Heatmap Visualization")
    print("=" * 60)
    
    # Initialize detector with attention enabled
    print("\n1. Initializing FireImageDetector with attention enabled...")
    detector = FireImageDetector(enable_attention=True)
    print("   ✓ Detector initialized")
    
    # Test with a fire image
    test_image_path = os.path.join(ROOT_DIR, "data", "raw", "images", "test_fire.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"\n   ⚠ Test image not found: {test_image_path}")
        print("   Skipping test...")
        return
    
    print(f"\n2. Analyzing test image: {test_image_path}")
    result = detector.analyze_image(test_image_path)
    
    if result is None:
        print("   ✗ Analysis failed - returned None")
        return
    
    print("   ✓ Analysis completed")
    
    # Check result structure
    print("\n3. Checking result structure...")
    if isinstance(result, dict):
        print("   ✓ Result is a dictionary")
        
        if 'results' in result:
            print(f"   ✓ 'results' key present with {len(result['results'])} classifications")
            top_result = result['results'][0]
            print(f"      - Top prediction: {top_result['label']} ({top_result['confidence']:.2f}%)")
        else:
            print("   ✗ 'results' key missing")
        
        if 'attention_heatmap' in result:
            heatmap = result['attention_heatmap']
            if heatmap is not None:
                print(f"   ✓ Attention heatmap generated: shape {heatmap.shape}, dtype {heatmap.dtype}")
                print(f"      - Min value: {heatmap.min()}, Max value: {heatmap.max()}")
            else:
                print("   ⚠ Attention heatmap is None")
                if 'error' in result and result['error']:
                    print(f"      Error: {result['error']}")
        else:
            print("   ✗ 'attention_heatmap' key missing")
    else:
        print(f"   ⚠ Result is not a dictionary (type: {type(result)})")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_attention_heatmap()
