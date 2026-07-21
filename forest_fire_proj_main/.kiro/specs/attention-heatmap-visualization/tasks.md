# Implementation Plan: Attention Heatmap Visualization

## Overview

This implementation plan breaks down the attention heatmap visualization feature into discrete coding tasks. The feature extracts attention weights from the Vision Transformer (ViT) model and generates visual heatmap overlays showing which image regions the model focuses on during fire detection. Implementation follows an incremental approach: create core components, integrate with existing detector, update UI, and add comprehensive testing.

## Tasks

- [x] 1. Set up project dependencies and core structure
  - Add new dependencies to requirements.txt: matplotlib>=3.5.0, opencv-python>=4.5.0, scipy>=1.7.0
  - Create new file `src/image_module/attention_heatmap.py` with module docstring
  - Set up logging configuration for attention heatmap module
  - _Requirements: 6.5_

- [x] 2. Implement AttentionHeatmapGenerator class - Core extraction methods
  - [x] 2.1 Create AttentionHeatmapGenerator class with __init__ method
    - Implement constructor accepting colormap (str) and opacity (float) parameters
    - Initialize colormap cache dictionary for performance optimization
    - Add input validation for colormap names ('viridis', 'plasma', 'inferno')
    - Add input validation for opacity range (0.4-0.6)
    - _Requirements: 2.2, 6.3_

  - [x] 2.2 Implement _aggregate_attention_heads method
    - Accept attention tensor with shape (num_heads, seq_len, seq_len)
    - Compute mean across attention heads (dim=0)
    - Return aggregated attention with shape (seq_len, seq_len)
    - _Requirements: 1.2_

  - [x] 2.3 Implement _extract_cls_attention method
    - Extract CLS token attention to image patches (first row, excluding CLS column)
    - Return 1D array of attention values for image patches only
    - _Requirements: 1.1, 1.3_

  - [x] 2.4 Implement _reshape_to_spatial method
    - Accept 1D attention array and patch_size parameter
    - Calculate grid dimensions: sqrt(num_patches)
    - Reshape to 2D spatial grid (grid_h, grid_w)
    - Handle non-square grids appropriately
    - _Requirements: 1.3_

  - [ ]* 2.5 Write property test for attention extraction pipeline
    - **Property 1: Attention Extraction Pipeline Produces Valid Spatial Map**
    - **Validates: Requirements 1.1, 1.2, 1.3**
    - Use hypothesis to generate random attention tensors with varying heads and sequence lengths
    - Verify output dimensions match expected patch grid size
    - Verify all values are finite (no NaN or Inf)

- [x] 3. Implement AttentionHeatmapGenerator class - Visualization methods
  - [x] 3.1 Implement _normalize_attention method
    - Accept 2D attention array
    - Normalize values to [0, 1] range using min-max normalization
    - Handle edge cases: all zeros, all same value, negative values
    - _Requirements: 2.4_

  - [x] 3.2 Implement _upsample_to_image_size method
    - Accept 2D attention map and target image dimensions (H, W)
    - Use cv2.resize with INTER_LINEAR interpolation for bilinear upsampling
    - Return upsampled attention map matching image dimensions
    - _Requirements: 2.1, 5.2_

  - [x] 3.3 Implement _apply_colormap method
    - Accept normalized attention values [0, 1]
    - Retrieve or create matplotlib colormap object (cache for reuse)
    - Apply colormap to convert attention to RGB
    - Convert to uint8 format [0, 255]
    - _Requirements: 2.2, 6.3_

  - [x] 3.4 Implement _blend_with_image method
    - Accept colored heatmap and original image (both RGB uint8)
    - Blend using weighted average: overlay = alpha * heatmap + (1-alpha) * image
    - Ensure output values remain in [0, 255] range
    - Return blended overlay as uint8 array
    - _Requirements: 2.3, 2.5_

  - [ ]* 3.5 Write property test for heatmap generation
    - **Property 2: Heatmap Generation Produces Valid RGB Overlay**
    - **Validates: Requirements 2.1, 2.2, 2.4**
    - Use hypothesis to generate random attention maps and images
    - Verify output has same dimensions as input image
    - Verify all pixel values in [0, 255] range
    - Verify normalized attention values in [0, 1] range

  - [ ]* 3.6 Write property test for heatmap blending
    - **Property 3: Heatmap Blending Preserves Image Dimensions**
    - **Validates: Requirements 2.3, 2.5**
    - Use hypothesis to generate random heatmaps, images, and opacity values
    - Verify output dimensions match input dimensions
    - Verify all pixel values remain valid [0, 255]

- [x] 4. Implement AttentionHeatmapGenerator class - Main generation method
  - [x] 4.1 Implement generate_heatmap method
    - Accept attention_weights (torch.Tensor), original_image (np.ndarray), patch_size (int)
    - Call extraction pipeline: aggregate → extract CLS → reshape → upsample
    - Call visualization pipeline: normalize → colormap → blend
    - Return final heatmap overlay as numpy array (H, W, 3)
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 4.2 Add comprehensive error handling to generate_heatmap
    - Wrap extraction pipeline in try-except for ValueError (invalid dimensions)
    - Wrap visualization pipeline in try-except for RuntimeError (memory errors)
    - Implement low-resolution fallback for memory errors
    - Log errors with appropriate severity levels (ERROR, WARNING)
    - Return None on unrecoverable errors
    - _Requirements: 5.1, 5.3, 5.4, 5.5_

  - [ ]* 4.3 Write unit tests for AttentionHeatmapGenerator
    - Test _aggregate_attention_heads with single head and multiple heads
    - Test _normalize_attention with edge cases (all zeros, all same, negative)
    - Test _reshape_to_spatial with various patch sizes
    - Test error handling when invalid inputs provided
    - _Requirements: 1.2, 2.4, 5.1_

- [x] 5. Checkpoint - Verify AttentionHeatmapGenerator works in isolation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Modify FireImageDetector class to extract attention weights
  - [x] 6.1 Update FireImageDetector.__init__ method
    - Add enable_attention parameter (default True)
    - Initialize AttentionHeatmapGenerator instance if attention enabled
    - Store enable_attention flag as instance variable
    - _Requirements: 4.5_

  - [x] 6.2 Update model loading to support attention extraction
    - Add attn_implementation="eager" parameter to from_pretrained call
    - This ensures attention weights are returned by the model
    - _Requirements: 1.1_

  - [x] 6.3 Modify analyze_image method to extract attention weights
    - Add output_attentions=True parameter to model forward pass
    - Extract attention weights from outputs.attentions (tuple of tensors)
    - Get final layer attention: outputs.attentions[-1]
    - Move attention tensor to CPU and convert to numpy if needed
    - _Requirements: 1.1, 4.1, 4.3, 6.2_

  - [x] 6.4 Integrate AttentionHeatmapGenerator into analyze_image workflow
    - After computing classification results, check if attention enabled
    - If enabled, call heatmap_generator.generate_heatmap with attention weights and image
    - Load original image as numpy array for heatmap generation
    - Handle case where heatmap generation returns None (error occurred)
    - _Requirements: 4.1, 4.2, 5.1_

  - [x] 6.5 Update analyze_image return value structure
    - Change return type from List[Dict] to Dict containing 'results' and 'attention_heatmap'
    - Maintain backward compatibility by keeping 'results' as list of classification dicts
    - Add 'attention_heatmap' key with numpy array or None
    - Add optional 'error' key if attention extraction failed
    - _Requirements: 4.2, 4.4_

  - [ ]* 6.6 Write property test for classification preservation
    - **Property 4: Classification Results Are Preserved During Attention Extraction**
    - **Validates: Requirements 1.5**
    - Run classification with and without attention enabled
    - Verify classification results (labels and confidence) are identical
    - Use mocked model outputs for deterministic testing

  - [ ]* 6.7 Write property test for single forward pass efficiency
    - **Property 6: Single Forward Pass Efficiency**
    - **Validates: Requirements 4.3, 6.2**
    - Mock model forward method to count invocations
    - Verify model is called exactly once per analyze_image call
    - Verify attention extracted from that single pass

  - [ ]* 6.8 Write unit tests for FireImageDetector modifications
    - Test analyze_image with attention enabled returns both results and heatmap
    - Test analyze_image with attention disabled returns only results
    - Test backward compatibility with existing result access patterns
    - Test error handling when attention extraction fails
    - _Requirements: 4.4, 4.5, 5.1_

- [x] 7. Checkpoint - Verify FireImageDetector integration works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Update AI Vision Lab UI to display attention heatmaps
  - [x] 8.1 Modify app.py to handle new analyze_image return structure
    - Update result handling to access results['results'] instead of results directly
    - Check for 'attention_heatmap' key in results dictionary
    - Handle case where attention_heatmap is None (not available)
    - _Requirements: 3.4, 4.2_

  - [x] 8.2 Add attention heatmap visualization section to UI
    - Add section header "🔍 Attention Heatmap" with divider
    - Add explanatory text describing heatmap meaning (warm = high attention, cool = low)
    - Create two-column layout for original image and heatmap overlay
    - Display original image in left column with "Original Image" label
    - Display attention heatmap in right column with "Attention Overlay" label
    - Use st.image with use_container_width=True for responsive display
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 8.3 Add conditional rendering for heatmap section
    - Only display heatmap section if 'attention_heatmap' exists and is not None
    - Maintain existing UI layout when heatmap unavailable
    - Display user-friendly message if attention extraction failed (check 'error' key)
    - _Requirements: 3.4, 5.5_

  - [ ]* 8.4 Write integration test for UI rendering
    - Test UI displays classification results correctly with new return structure
    - Test UI displays heatmap when available
    - Test UI handles missing heatmap gracefully
    - Test UI displays error message when attention extraction fails
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.5_

- [ ] 9. Add comprehensive property-based tests for remaining properties
  - [ ]* 9.1 Write property test for analysis return structure
    - **Property 5: Analysis Returns Both Classification and Attention Data**
    - **Validates: Requirements 4.1, 4.2**
    - Verify analyze_image returns dict with 'results' and 'attention_heatmap' keys
    - Verify 'results' is a list of classification dicts
    - Verify 'attention_heatmap' is numpy array or None

  - [ ]* 9.2 Write property test for backward compatibility
    - **Property 7: Backward Compatibility Maintained**
    - **Validates: Requirements 4.4**
    - Test existing access patterns: results['results'][0]['label']
    - Verify no errors when accessing classification data
    - Test with various classification result structures

  - [ ]* 9.3 Write property test for graceful degradation
    - **Property 8: Graceful Degradation When Attention Disabled**
    - **Validates: Requirements 4.5**
    - Initialize detector with enable_attention=False
    - Verify analyze_image returns classification results normally
    - Verify no attention_heatmap key or it is None
    - Verify no errors or exceptions raised

  - [ ]* 9.4 Write property test for unusual image dimensions
    - **Property 9: Unusual Image Dimensions Handled Correctly**
    - **Validates: Requirements 5.2**
    - Use hypothesis to generate images with extreme aspect ratios
    - Verify heatmap generation succeeds or fails gracefully
    - Verify output dimensions match input when successful

  - [ ]* 9.5 Write property test for GPU acceleration
    - **Property 10: GPU Acceleration Used When Available**
    - **Validates: Requirements 6.4**
    - Check if CUDA available using torch.cuda.is_available()
    - Verify attention tensors processed on GPU when available
    - Verify fallback to CPU when GPU unavailable

  - [ ]* 9.6 Write property test for colormap caching
    - **Property 11: Colormap Caching Across Multiple Images**
    - **Validates: Requirements 6.3**
    - Generate multiple heatmaps with same colormap configuration
    - Verify colormap object created only once
    - Verify cache hit rate increases with multiple generations

- [x] 10. Add performance optimizations and monitoring
  - [x] 10.1 Implement colormap caching mechanism
    - Store colormap objects in instance variable dictionary
    - Key by colormap name for quick lookup
    - Create colormap only on first use, reuse for subsequent calls
    - _Requirements: 6.3_

  - [x] 10.2 Add GPU memory management
    - Explicitly delete intermediate attention tensors after use
    - Move tensors to CPU before numpy conversion to free GPU memory
    - Use torch.cuda.empty_cache() after processing if GPU used
    - _Requirements: 6.4_

  - [x] 10.3 Add timing instrumentation for performance monitoring
    - Log heatmap generation time at INFO level
    - Track whether GPU or CPU was used for processing
    - Log warning if generation exceeds 2 second threshold
    - _Requirements: 6.1_

  - [ ]* 10.4 Write integration test for performance requirements
    - Test heatmap generation completes within 2 seconds for 2048x2048 images
    - Test with various image sizes (224x224, 512x512, 1024x1024, 2048x2048)
    - Verify GPU acceleration provides speedup when available
    - _Requirements: 6.1, 6.4_

- [x] 11. Final integration and end-to-end testing
  - [x] 11.1 Test complete workflow with real fire detection images
    - Use images from data/raw/training_images/fire directory
    - Verify heatmaps highlight fire regions appropriately
    - Verify classification results remain accurate
    - _Requirements: 1.1, 2.1, 3.1, 4.1_

  - [x] 11.2 Test complete workflow with non-fire images
    - Use images from data/raw/training_images/no_fire directory
    - Verify heatmaps show appropriate attention distribution
    - Verify classification results remain accurate
    - _Requirements: 1.1, 2.1, 3.1, 4.1_

  - [ ]* 11.3 Write integration test for error scenarios
    - Test with corrupted image file
    - Test with unsupported image format
    - Test with extremely small image (<50x50)
    - Test with extremely large image (>4096x4096)
    - Verify graceful error handling in all cases
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 12. Final checkpoint - Ensure all tests pass and feature is complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP delivery
- Each task references specific requirements from the requirements document for traceability
- Property-based tests use the `hypothesis` library with minimum 100 iterations per test
- Checkpoints ensure incremental validation at key milestones
- The implementation maintains backward compatibility with existing code
- Performance optimizations (colormap caching, GPU acceleration) are built into core implementation
- Error handling is comprehensive to ensure core fire detection functionality never breaks
