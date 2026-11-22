import os
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

class FireImageDetector:
    def __init__(self, enable_attention: bool = True):
        print("Loading YOUR Custom Vision Transformer (ViT)...")
        # Build the path to your locally trained model
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_name = os.path.join(ROOT_DIR, "models", "custom_vit_fire")
        
        # Load your custom processor and model weights
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            attn_implementation="eager"  # Required for attention extraction
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize attention heatmap generator if enabled
        self.enable_attention = enable_attention
        self.heatmap_generator = None
        if enable_attention:
            try:
                from src.image_module.attention_heatmap import AttentionHeatmapGenerator
                self.heatmap_generator = AttentionHeatmapGenerator()
                print("Attention heatmap visualization enabled")
            except ImportError as e:
                print(f"Warning: Could not load AttentionHeatmapGenerator: {e}")
                self.enable_attention = False

    def analyze_image(self, image_path):
        """Passes an image through the ViT and returns the predictions with optional attention heatmap."""
        if not os.path.exists(image_path):
            return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Run model with attention extraction if enabled
            with torch.no_grad():
                if self.enable_attention and self.heatmap_generator is not None:
                    outputs = self.model(**inputs, output_attentions=True)
                else:
                    outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            top_indices = probabilities.argsort(descending=True)
            
            results = []
            for idx in top_indices:
                # Your custom model uses your actual folder names as labels!
                raw_label = self.model.config.id2label[idx.item()]
                
                if raw_label == "fire":
                    label = "🔥 Fire Detected"
                else:
                    label = "🌲 No Fire"
                    
                confidence = probabilities[idx].item() * 100
                results.append({"label": label, "confidence": confidence})
            
            # Prepare return value with backward compatibility
            response = {
                'results': results,
                'attention_heatmap': None,
                'error': None
            }
            
            # Generate attention heatmap if enabled
            if self.enable_attention and self.heatmap_generator is not None:
                try:
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        # Extract final layer attention
                        final_attention = outputs.attentions[-1]  # Shape: (batch, heads, seq, seq)
                        final_attention = final_attention[0]  # Remove batch dimension
                        
                        # Convert image to numpy array for heatmap generation
                        image_np = np.array(image)
                        
                        # Generate heatmap
                        heatmap = self.heatmap_generator.generate_heatmap(
                            final_attention,
                            image_np,
                            patch_size=16
                        )
                        
                        response['attention_heatmap'] = heatmap
                        
                        if heatmap is None:
                            response['error'] = "Attention heatmap generation failed"
                    else:
                        response['error'] = "Model did not return attention weights"
                        
                except Exception as e:
                    response['error'] = f"Attention extraction error: {str(e)}"
                    print(f"Warning: Attention extraction failed: {e}")
            
            return response
                
        except Exception as e:
            print(f"Failed to process image: {e}")
            return None

# --- Quick Test Block ---
if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_image_path = os.path.join(ROOT_DIR, "data", "raw", "images", "test_fire.jpg")
    
    detector = FireImageDetector()
    results = detector.analyze_image(test_image_path)
    if results:
        for res in results:
            print(f" - {res['label']}: {res['confidence']:.2f}%")