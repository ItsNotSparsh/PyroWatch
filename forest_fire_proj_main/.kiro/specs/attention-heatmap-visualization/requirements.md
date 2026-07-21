# Requirements Document

## Introduction

This document specifies requirements for adding attention heatmap visualization to the PyroWatch wildfire detection application. The feature will provide visual explainability by showing which regions of uploaded images the Vision Transformer (ViT) model focuses on when making fire detection decisions. This enhances user trust and understanding of AI predictions by making the model's decision-making process transparent.

## Glossary

- **Attention_Heatmap_Generator**: Component that extracts attention weights from the ViT model and generates visual heatmaps
- **ViT_Model**: Vision Transformer deep learning model (custom_vit_fire) used for fire detection
- **Attention_Weights**: Numerical values representing which image regions the model focuses on during classification
- **Heatmap_Overlay**: Visual representation of attention weights superimposed on the original image
- **AI_Vision_Lab**: The second tab (tab2) in the Streamlit application where image analysis occurs
- **FireImageDetector**: Existing class in src/image_module/vit_model.py that handles image analysis
- **Classification_Result**: The fire detection prediction (Fire Detected or No Fire) with confidence score

## Requirements

### Requirement 1: Extract Attention Weights from ViT Model

**User Story:** As a wildfire analyst, I want the system to extract attention information from the ViT model, so that I can understand which image regions influenced the fire detection decision.

#### Acceptance Criteria

1. WHEN an image is analyzed, THE Attention_Heatmap_Generator SHALL extract attention weights from the final attention layer of the ViT_Model
2. THE Attention_Heatmap_Generator SHALL aggregate attention weights across all attention heads in the final layer
3. THE Attention_Heatmap_Generator SHALL reshape attention weights to match the spatial dimensions of the input image patches
4. WHEN attention extraction fails, THE Attention_Heatmap_Generator SHALL return a descriptive error message
5. THE Attention_Heatmap_Generator SHALL preserve the original Classification_Result during attention extraction

### Requirement 2: Generate Visual Heatmap Overlay

**User Story:** As a wildfire analyst, I want to see a color-coded heatmap overlaid on the original image, so that I can visually identify which regions the AI is analyzing.

#### Acceptance Criteria

1. THE Attention_Heatmap_Generator SHALL convert attention weights to a 2D heatmap matching the original image dimensions
2. THE Attention_Heatmap_Generator SHALL apply a perceptually-uniform colormap (viridis, plasma, or inferno) to the heatmap
3. THE Attention_Heatmap_Generator SHALL blend the heatmap with the original image at 40-60% opacity for visibility
4. THE Attention_Heatmap_Generator SHALL normalize attention values to the range [0, 1] before colormap application
5. WHEN generating the overlay, THE Attention_Heatmap_Generator SHALL maintain the original image aspect ratio and resolution

### Requirement 3: Display Heatmap in AI Vision Lab Interface

**User Story:** As a wildfire analyst, I want to see the attention heatmap displayed alongside the classification results, so that I can evaluate the model's reasoning in context.

#### Acceptance Criteria

1. WHEN image analysis completes, THE AI_Vision_Lab SHALL display the Heatmap_Overlay below the classification results
2. THE AI_Vision_Lab SHALL display the original image and Heatmap_Overlay side-by-side for comparison
3. THE AI_Vision_Lab SHALL include a legend explaining the heatmap color scale (cool colors = low attention, warm colors = high attention)
4. WHEN no heatmap is available, THE AI_Vision_Lab SHALL display only the classification results without error
5. THE AI_Vision_Lab SHALL maintain the existing UI styling and layout consistency

### Requirement 4: Integrate with Existing Analysis Workflow

**User Story:** As a wildfire analyst, I want attention heatmaps to be generated automatically during image analysis, so that I don't need to perform additional steps.

#### Acceptance Criteria

1. WHEN the "Run Deep Learning Analysis" button is clicked, THE FireImageDetector SHALL generate both Classification_Result and Attention_Weights
2. THE FireImageDetector SHALL return attention data alongside classification results in a single response
3. WHEN the ViT_Model processes an image, THE FireImageDetector SHALL extract attention weights without additional forward passes
4. THE FireImageDetector SHALL maintain backward compatibility with existing code that only uses Classification_Result
5. WHEN attention extraction is disabled, THE FireImageDetector SHALL continue to provide Classification_Result normally

### Requirement 5: Handle Edge Cases and Errors

**User Story:** As a wildfire analyst, I want the system to handle errors gracefully, so that attention visualization failures don't break the core fire detection functionality.

#### Acceptance Criteria

1. IF attention extraction fails, THEN THE FireImageDetector SHALL log the error and return Classification_Result without attention data
2. WHEN the uploaded image has unusual dimensions, THE Attention_Heatmap_Generator SHALL resize the heatmap appropriately
3. IF the ViT_Model architecture doesn't support attention extraction, THEN THE Attention_Heatmap_Generator SHALL return a clear error message
4. WHEN memory constraints prevent heatmap generation, THE Attention_Heatmap_Generator SHALL fall back to lower resolution visualization
5. THE AI_Vision_Lab SHALL display a user-friendly message when attention visualization is unavailable

### Requirement 6: Optimize Performance

**User Story:** As a wildfire analyst, I want attention heatmap generation to be fast, so that it doesn't significantly slow down image analysis.

#### Acceptance Criteria

1. THE Attention_Heatmap_Generator SHALL generate heatmaps within 2 seconds for images up to 2048x2048 pixels
2. THE Attention_Heatmap_Generator SHALL reuse existing model outputs without requiring additional inference
3. WHEN processing multiple images, THE Attention_Heatmap_Generator SHALL cache colormap computations for reuse
4. THE FireImageDetector SHALL use GPU acceleration for attention weight processing when available
5. THE Attention_Heatmap_Generator SHALL use efficient numpy operations for array manipulations
