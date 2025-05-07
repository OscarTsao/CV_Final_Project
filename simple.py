import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os
from tqdm import tqdm
import json

# Step 1: Image Reading and Label Extraction
# Load all images and labels from subfolders
data_path = 'data/train'  # Replace with the actual dataset path
categories = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
label_map = {category: idx for idx, category in enumerate(categories)}

images = []
labels = []
print("Reading images...")
for category in tqdm(categories, desc="Processing Categories"):
    folder_path = os.path.join(data_path, category)
    for file in tqdm(glob.glob(os.path.join(folder_path, '*.jpg')), desc=f"Loading {category}", leave=False):
        image = cv2.imread(file)
        images.append(image)
        labels.append(label_map[category])

# Step 2: Image Preprocessing
# Resize images, convert to grayscale, apply thresholding
processed_images = []
threshold_value = 120
resize_shape = (45, 45)
print("Preprocessing images...")
for img in tqdm(images, desc="Resizing and Thresholding"):
    resized_img = cv2.resize(img, resize_shape)  # Resize to 45x45
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_TOZERO)
    processed_images.append(thresholded_img)

# Convert images to NumPy array and normalize
print("Converting and normalizing images...")
processed_images = np.array(processed_images) / 255.0  # Normalize pixel values
processed_images = processed_images.reshape(-1, 1, 45, 45)  # Add channel dimension for CNN
labels = np.array(labels)

# Step 3: Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    processed_images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Step 4: Oversampling with SMOTE
print("Applying SMOTE to balance the training data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten for SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
X_train_resampled = X_train_resampled.reshape(-1, 1, 45, 45)  # Reshape back

# Convert to PyTorch tensors
print("Converting data to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 5: CNN Model Definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultilayerModel(nn.Module):
    def __init__(self, input_dim=2025):  
        """
        Args:
            input_dim (int): Number of input features.
        """
        super(SimpleMultilayerModel, self).__init__()

        # Input layer: Dense (256) + ReLU
        self.layer_in = nn.Linear(input_dim, 256)

        # Hidden 1: Dropout(0.05)
        self.dropout1 = nn.Dropout(p=0.05)

        # Hidden 2: Dense (128) + ReLU
        self.layer2 = nn.Linear(256, 128)

        # Hidden 3: Dropout(0.05)
        self.dropout2 = nn.Dropout(p=0.05)

        # Hidden 4: Dense (128) + ReLU
        self.layer3 = nn.Linear(128, 128)

        # Hidden 5: Dropout(0.05)
        self.dropout3 = nn.Dropout(p=0.05)

        # Hidden 6: Dense (128) + ReLU
        self.layer4 = nn.Linear(128, 128)

        # Hidden 7: Dropout(0.05)
        self.dropout4 = nn.Dropout(p=0.05)

        # Hidden 8: Dense (128) + ReLU
        self.layer5 = nn.Linear(128, 128)

        # Hidden 9: Dropout(0.05)
        self.dropout5 = nn.Dropout(p=0.05)

        # Hidden 10: Dense (64) + ReLU
        self.layer6 = nn.Linear(128, 64)

        # Hidden 11: Dropout(0.05)
        self.dropout6 = nn.Dropout(p=0.05)

        # Hidden 12: Dense (64) + ReLU
        self.layer7 = nn.Linear(64, 64)

        # Hidden 13: Dropout(0.05)
        self.dropout7 = nn.Dropout(p=0.05)

        # Output layer: Dense (4) + Softmax
        self.layer_out = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape is [batch_size, channels, height, width]
    # Flatten
        x = x.view(x.size(0), -1)  # => [batch_size, input_dim]
        # Input layer
        x = F.relu(self.layer_in(x))

        # Hidden layers follow the pattern: Dropout -> Dense + ReLU -> Dropout -> ...
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.relu(self.layer4(x))
        x = self.dropout4(x)
        x = F.relu(self.layer5(x))
        x = self.dropout5(x)
        x = F.relu(self.layer6(x))
        x = self.dropout6(x)
        x = F.relu(self.layer7(x))
        x = self.dropout7(x)

        # Output layer
        x = self.layer_out(x)
        x = self.softmax(x)

        return x


# Step 6: Model Training and Evaluation
def train_model(model, train_loader, test_loader, epochs=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    metrics = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

        # Calculate metrics
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    metrics['train_loss'] = train_losses
    metrics['val_loss'] = val_losses
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    metrics['accuracy'] = accuracy

    # Save metrics to JSON
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

# Main execution
if __name__ == "__main__":
    print("Initializing model...")
    model = SimpleMultilayerModel()
    train_model(model, train_loader, test_loader, epochs=25, lr=0.001)
