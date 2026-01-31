import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# TASK REQUIREMENT: VGG (Vision) + LSTM (NLP)
# ==========================================

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_size=256, hidden_size=256, max_length=35):
        super(ImageCaptioningModel, self).__init__()
        
        # --- 1. ENCODER (Image Features - VGG16) ---
        print("Loading VGG16 Encoder...")
        vgg = models.vgg16(pretrained=True)
        # Remove classification layer (classifier[6] is usually the last one)
        # We want the 4096 output from classifier[0] or similar features
        # For simplicity in PyTorch, we can use the features + avgpool
        modules = list(vgg.children())[:-1] 
        self.vgg = nn.Sequential(*modules)
        
        # Linear layer to transform image features to embedding size
        # VGG16 output is (512, 7, 7) -> flattened is 25088 for full features
        # Or we can use the classifier part. Let's assume a simplified feature extraction.
        self.fc_image = nn.Linear(512 * 7 * 7, embed_size) 
        self.dropout_img = nn.Dropout(0.5)

        # --- 2. DECODER (Text Generation - LSTM) ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout_text = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # --- 3. OUTPUT ---
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # Image Features
        features = self.vgg(images)
        features = features.view(features.size(0), -1) # Flatten
        features = self.fc_image(features)
        features = self.dropout_img(features)
        features = features.unsqueeze(1) # Add sequence dim

        # Caption Embeddings
        embeddings = self.embedding(captions)
        embeddings = self.dropout_text(embeddings)

        # Concatenate: [Image, Word1, Word2...]
        inputs = torch.cat((features, embeddings), dim=1)
        
        # LSTM
        hiddens, _ = self.lstm(inputs)
        
        # Output
        outputs = self.fc_out(hiddens)
        return outputs

if __name__ == "__main__":
    print("Building Image Captioning Model (VGG16 + LSTM) in PyTorch...")
    try:
        model = ImageCaptioningModel()
        print("\nSUCCESS: Model Architecture defined.")
        print(model)
        print("\nNote: This is the raw architecture. To use it, it must be trained on a dataset like COCO or Flickr8k.")
    except Exception as e:
        print(f"Error building model: {e}")
