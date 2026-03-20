"""
====================================================================
CONVNEXT-TINY + COSINE EMBEDDINGS + kNN PIPELINE
Fashion-MNIST Style Dataset

Pipeline:
    Dataset → ConvNeXt → Embeddings →
    kNN Selection → Metrics →
    Diagnostics → Export Results
====================================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

# Numerical operations
import numpy as np

# Data handling
import pandas as pd

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset utilities
from torch.utils.data import Dataset, DataLoader

# Pretrained vision models
from torchvision import models, transforms

# kNN + evaluation metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

# Multiclass ROC preparation
from sklearn.preprocessing import label_binarize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Timing execution
import time

# Google Colab downloader
from google.colab import files


# ==========================================================
# 0. REPRODUCIBILITY
# ==========================================================

SEED = 42                      # fixed seed for experiment repeatability
np.random.seed(SEED)           # numpy randomness control
torch.manual_seed(SEED)        # torch randomness control

# Select GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================================
# 1. DATA LOADING
# ==========================================================

"""
Uploads CSV dataset from local machine.
"""

uploaded = files.upload()

# Load datasets
train_df = pd.read_csv("product_images.csv")
test_df  = pd.read_csv("product_images_for_prediction.csv")

# Separate features and labels
X_train = train_df.drop("label", axis=1).values.astype(np.float32)
y_train = train_df["label"].values

X_test = test_df.values.astype(np.float32)

# Normalize pixel range [0,255] → [0,1]
X_train /= 255.0
X_test  /= 255.0

# Reshape flat vectors → image tensors
X_train = X_train.reshape(-1,1,28,28)
X_test  = X_test.reshape(-1,1,28,28)

print("Training shape:", X_train.shape)


# ==========================================================
# 2. DATASET CLASS
# ==========================================================

# Resize transform required because ConvNeXt expects 224x224 inputs
resize_transform = transforms.Resize((224, 224))


class FashionDataset(Dataset):
    """
    Custom PyTorch Dataset used for loading Fashion-MNIST style image data
    into a format compatible with deep learning pipelines and pretrained
    ConvNeXt feature extractors.

    This dataset acts as an interface between raw NumPy image arrays and
    PyTorch DataLoader objects.

    The class performs automatic tensor conversion, spatial resizing,
    and optional label retrieval during indexing.

    Args:
        X (np.ndarray):
            Array containing image data.
            Expected shape:
                (N, C, H, W)
            where:
                N = number of samples
                C = number of channels
                H = height
                W = width

        y (np.ndarray | None, optional):
            Corresponding class labels.
            If None, dataset operates in inference mode.
            Default = None.

    Behavior:
        • Converts NumPy arrays into torch tensors.
        • Stores labels only if provided.
        • Resizes images dynamically during retrieval.
        • Supports supervised and unsupervised workflows.
        • Compatible with PyTorch batching mechanisms.

    Returns:
        tuple(torch.Tensor, torch.Tensor):
            Returned when labels exist.
            (image_tensor, label_tensor)

        torch.Tensor:
            Returned when labels are absent.
            image_tensor only.

    Raises:
        IndexError:
            If requested index exceeds dataset size.

        RuntimeError:
            If tensor conversion or resizing fails.
    """

    def __init__(self, X, y=None):
        """
        Initializes dataset storage.

        Args:
            X (np.ndarray):
                Input image array.

            y (np.ndarray | None):
                Optional labels.

        Behavior:
            Converts images into tensors and determines whether
            labels should be returned during indexing.

        Output:
            None
        """

        # Convert NumPy images into PyTorch tensor format
        self.X = torch.tensor(X)

        # Store labels directly without modification
        self.y = y

        # Boolean flag determining supervised vs inference mode
        self.has_label = y is not None

        # Print dataset initialization diagnostics
        print(f"[Dataset Init] Samples loaded: {len(self.X)}")

        # Print whether labels are available
        print(f"[Dataset Init] Labels present: {self.has_label}")

    def __len__(self):
        """
        Returns total dataset size.

        Behavior:
            Enables PyTorch DataLoader to determine iteration count.

        Returns:
            int:
                Number of samples contained in dataset.
        """

        # Return number of stored samples
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a single dataset sample.

        Args:
            idx (int):
                Index of requested sample.

        Behavior:
            • Fetches image tensor.
            • Applies resize transform.
            • Returns label if available.
            • Provides verbose execution tracing.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                Returned when labels exist.

            torch.Tensor:
                Returned when labels do not exist.

        Raises:
            IndexError:
                If index is outside dataset bounds.

            Exception:
                Any transformation failure is caught and reported.
        """

        try:
            # Print accessed index for debugging traceability
            print(f"[Dataset] Fetching index: {idx}")

            # Retrieve raw tensor image
            img = self.X[idx]

            # Resize image to ConvNeXt-compatible resolution
            img = resize_transform(img)

            # Check whether dataset contains labels
            if self.has_label:

                # Convert label into tensor format
                label = torch.tensor(self.y[idx])

                # Print label retrieval confirmation
                print(f"[Dataset] Returning labeled sample")

                return img, label

            # Print inference-mode retrieval
            print(f"[Dataset] Returning unlabeled sample")

            return img

        except IndexError as e:
            # Explicit index failure reporting
            print(f"[ERROR] Invalid index requested: {idx}")
            raise e

        except Exception as e:
            # Catch unexpected transformation issues
            print("[ERROR] Dataset retrieval failure")
            raise RuntimeError(e)


# ==========================================================
# 3. MODEL
# ==========================================================

class ConvNeXtFineTune(nn.Module):
    """
    ConvNeXt Tiny based neural network used for feature extraction
    and downstream classification tasks on grayscale image datasets.

    This model adapts a pretrained ConvNeXt architecture to accept
    single-channel inputs while producing normalized embeddings
    suitable for similarity learning or kNN classification.

    Args:
        embedding_dim (int, optional):
            Dimensionality of learned embedding space.
            Controls representation compactness.
            Default = 256.

        num_classes (int, optional):
            Number of classification categories.
            Determines classifier output size.
            Default = 10.

    Behavior:
        • Converts grayscale images into RGB format.
        • Utilizes pretrained ConvNeXt Tiny backbone.
        • Performs global spatial pooling.
        • Projects features into embedding space.
        • Normalizes embeddings using L2 normalization.
        • Optionally returns embeddings instead of logits.
        • Provides verbose execution diagnostics.

    Forward Pass Pipeline:
        Input → Channel Expansion →
        ConvNeXt Feature Extraction →
        Global Average Pool →
        Flatten →
        Linear Embedding Projection →
        L2 Normalization →
        Classification Layer (optional)

    Returns:
        torch.Tensor:
            Classification logits when return_embedding=False.

        torch.Tensor:
            Normalized embedding vectors when
            return_embedding=True.

    Raises:
        RuntimeError:
            If tensor dimensions are incompatible.

        Exception:
            Any unexpected forward-pass failure.
    """

    def __init__(self,
                 embedding_dim=256,
                 num_classes=10):
        """
        Initializes ConvNeXt fine-tuning architecture.

        Args:
            embedding_dim (int):
                Desired embedding size.

            num_classes (int):
                Total prediction classes.

        Behavior:
            Loads pretrained ConvNeXt Tiny backbone and
            replaces classification head with embedding
            projection and custom classifier.

        Output:
            None
        """

        # Initialize parent PyTorch module
        super().__init__()

        print("[Model Init] Loading ConvNeXt Tiny backbone")

        # Load pretrained ConvNeXt Tiny model
        base = models.convnext_tiny(weights="DEFAULT")

        # Convert grayscale input channel to RGB
        self.input_conv = nn.Conv2d(1, 3, 1)

        # Extract convolutional feature backbone
        self.features = base.features

        # Perform spatial global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Retrieve feature dimension from pretrained classifier
        in_features = base.classifier[2].in_features

        print(f"[Model Init] Backbone feature size: {in_features}")

        # Linear layer mapping backbone features to embedding space
        self.embedding = nn.Linear(
            in_features,
            embedding_dim
        )

        # Final classification layer operating on embeddings
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

        print(f"[Model Init] Embedding dimension: {embedding_dim}")
        print(f"[Model Init] Number of classes: {num_classes}")

    def forward(self, x, return_embedding=False):
        """
        Executes forward propagation through network.

        Args:
            x (torch.Tensor):
                Input image batch.
                Expected shape:
                    (batch_size, 1, H, W)

            return_embedding (bool, optional):
                If True, skips classifier and returns
                normalized embeddings.
                Default = False.

        Behavior:
            • Expands grayscale channels.
            • Extracts ConvNeXt features.
            • Applies adaptive pooling.
            • Flattens feature maps.
            • Generates embeddings.
            • Applies L2 normalization.
            • Optionally computes logits.

        Returns:
            torch.Tensor:
                Normalized embeddings OR classification logits.

        Raises:
            RuntimeError:
                If invalid tensor dimensions occur.

            Exception:
                Any unexpected computation failure.
        """

        try:
            print("[Forward] Starting forward pass")

            # Convert grayscale input into 3-channel representation
            x = self.input_conv(x)

            print(f"[Forward] After RGB conversion: {x.shape}")

            # Pass data through ConvNeXt feature extractor
            x = self.features(x)

            print(f"[Forward] Feature map shape: {x.shape}")

            # Apply global spatial pooling
            x = self.avgpool(x)

            print(f"[Forward] After pooling: {x.shape}")

            # Flatten pooled tensor into vector representation
            x = torch.flatten(x, 1)

            print(f"[Forward] After flattening: {x.shape}")

            # Project backbone features into embedding space
            emb = self.embedding(x)

            print(f"[Forward] Raw embedding shape: {emb.shape}")

            # Normalize embeddings for cosine similarity usage
            emb = F.normalize(emb, dim=1)

            print("[Forward] Embeddings normalized")

            # Return embeddings if requested
            if return_embedding:
                print("[Forward] Returning embeddings only")
                return emb

            # Compute classification logits
            logits = self.classifier(emb)

            print(f"[Forward] Logits shape: {logits.shape}")

            return logits

        except RuntimeError as e:
            print("[ERROR] Tensor dimension mismatch during forward pass")
            raise e

        except Exception as e:
            print("[ERROR] Unexpected forward pass failure")
            raise RuntimeError(e)

# ==========================================================
# 4. TRAINING SETUP
# ==========================================================

# Create DataLoader for batching and shuffling training data
train_loader = DataLoader(
    FashionDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

# Initialize ConvNeXt fine-tuning model on selected device
model = ConvNeXtFineTune().to(device)

# Freeze pretrained ConvNeXt backbone parameters
for p in model.features.parameters():
    p.requires_grad = False

# Define AdamW optimizer for stable transformer-style training
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2
)

# Apply cosine annealing learning-rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=8
)

# Use cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()


# ==========================================================
# 5. TRAINING LOOP
# ==========================================================

# Display training start message
print("\nTraining ConvNeXt...")

# Record training start time
start_time = time.time()

# Define total number of training epochs
EPOCHS = 8

# Iterate through training epochs
for epoch in range(EPOCHS):

    # Gradually unfreeze backbone after warmup phase
    if epoch == 2:
        print("Unfreezing backbone...")
        for p in model.features.parameters():
            p.requires_grad = True

    # Set model to training mode
    model.train()

    # Initialize epoch loss accumulator
    total_loss = 0

    # Iterate over mini-batches
    for xb, yb in train_loader:

        # Move batch data to computation device
        xb, yb = xb.to(device), yb.to(device)

        # Reset gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass through network
        logits = model(xb)

        # Compute classification loss
        loss = criterion(logits, yb)

        # Backpropagate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate batch loss
        total_loss += loss.item()

    # Update learning rate using scheduler
    scheduler.step()

    # Print epoch training progress
    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"| Loss {total_loss:.3f}"
    )

# Print total training duration
print("Training time:",
      round(time.time()-start_time,2),"sec")


# ==========================================================
# 6. EMBEDDING EXTRACTION
# ==========================================================

def extract_embeddings(model, X):
    """
    Extract normalized feature embeddings from images using ConvNeXt.

    Args:
        model (ConvNeXtFineTune):
            Trained ConvNeXt model capable of returning embeddings.

        X (np.ndarray):
            Image tensor array of shape (N, C, H, W).

    Behavior:
        • Wraps input images into FashionDataset
        • Processes images in batches using DataLoader
        • Disables gradient computation for efficiency
        • Performs forward pass in embedding mode
        • Collects embeddings from all batches
        • Converts embeddings to NumPy format

    Returns:
        np.ndarray:
            Array of normalized embeddings with shape (N, embedding_dim).
    """

    # Create DataLoader for sequential embedding extraction
    loader = DataLoader(
        FashionDataset(X),
        batch_size=128,
        shuffle=False
    )

    # Switch model to evaluation mode
    model.eval()

    # Initialize container for batch embeddings
    embeddings = []

    # Disable gradient tracking during inference
    with torch.no_grad():

        # Iterate through dataset batches
        for xb in loader:

            # Move batch to computation device
            xb = xb.to(device)

            # Forward pass requesting embeddings instead of logits
            emb = model(
                xb,
                return_embedding=True
            )

            # Move embeddings to CPU memory
            embeddings.append(emb.cpu())

    # Concatenate all batches and convert to NumPy array
    return torch.cat(embeddings).numpy()


# Display embedding extraction status
print("\nExtracting embeddings...")

# Extract embeddings for training data
X_train_emb = extract_embeddings(model, X_train)

# Extract embeddings for test data
X_test_emb  = extract_embeddings(model, X_test)


# ==========================================================
# 7. kNN SELECTION
# ==========================================================

# Store optimal k value
best_k = None

# Track best achieved accuracy
best_score = 0

# Store probability outputs of best model
preds_proba_final = None

# Container for evaluation metrics across k values
results_k = []

# Evaluate multiple neighborhood sizes
for k in [3,5,7,9,11]:

    # Initialize cosine-distance kNN (+1 for self neighbor)
    knn = KNeighborsClassifier(
        n_neighbors=k+1,
        metric="cosine"
    )

    # Fit kNN using learned embeddings
    knn.fit(X_train_emb, y_train)

    # Compute nearest neighbors for training samples
    _, indices = knn.kneighbors(X_train_emb)

    # Remove self-neighbor to simulate LOOCV
    neighbor_labels = y_train[
        indices[:,1:k+1]
    ]

    # Perform majority voting across neighbors
    preds = np.array([
        np.bincount(
            row,
            minlength=10
        ).argmax()
        for row in neighbor_labels
    ])

    # Initialize probability matrix
    preds_proba = np.zeros(
        (len(y_train),10)
    )

    # Estimate class probabilities via neighbor frequency
    for c in range(10):
        preds_proba[:,c] = (
            neighbor_labels == c
        ).mean(axis=1)

    # Compute classification accuracy
    acc = accuracy_score(y_train, preds)

    # Compute macro-averaged precision
    prec = precision_score(
        y_train, preds,
        average='macro'
    )

    # Compute macro-averaged recall
    rec = recall_score(
        y_train, preds,
        average='macro'
    )

    # Convert labels into one-hot encoding
    y_onehot = np.eye(10)[y_train]

    # Compute One-vs-Rest ROC-AUC score
    roc_auc = roc_auc_score(
        y_onehot,
        preds_proba,
        multi_class='ovr'
    )

    # Store evaluation metrics for current k
    results_k.append({
        "k": k,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc
    })

    # Print performance summary for monitoring
    print(
        f"k={k} | Acc={acc:.4f} "
        f"| Precision={prec:.4f} "
        f"| Recall={rec:.4f} "
        f"| ROC-AUC={roc_auc:.4f}"
    )

    # Update best configuration if accuracy improves
    if acc > best_score:
        best_score = acc
        best_k = k
        preds_proba_final = preds_proba.copy()

# Display selected optimal k value
print("Best k:", best_k)

# ==========================================================
# 8. DIAGNOSTIC VISUALIZATION UTILITIES
# ==========================================================

def download_table(df: pd.DataFrame, filename: str):
    """
    Export a dataframe as CSV and automatically download it in Colab.

    Args:
        df (pd.DataFrame):
            Structured tabular data containing experiment results,
            evaluation metrics, or diagnostic summaries to be saved.

        filename (str):
            Target filename including extension (.csv) used for
            persistent storage and browser download.

    Behavior:
        • Writes dataframe safely to local runtime storage.
        • Initiates Google Colab download interaction.
        • Logs execution progress for transparency.
        • Handles runtime I/O failures gracefully.

    Returns:
        None:
            Function performs side-effects only (file creation + download).

    Raises:
        Exception:
            Captured internally and printed without interrupting pipeline.
    """

    try:
        # Notify user export process has started
        print(f"[INFO] Saving table -> {filename}")

        # Persist dataframe into runtime filesystem
        df.to_csv(filename, index=False)

        # Trigger automatic browser-side download
        files.download(filename)

        # Confirm successful export operation
        print("[SUCCESS] Table downloaded.")

    except Exception as e:
        # Catch filesystem or Colab download failures
        print("[ERROR] Table export failed:", e)



def save_and_download_plot(filename: str):
    """
    Save current matplotlib figure and download it locally.

    Args:
        filename (str):
            Output image filename including extension
            (e.g., '.png', '.jpg').

    Behavior:
        • Saves currently active matplotlib figure.
        • Initiates Colab download automatically.
        • Frees matplotlib memory after saving.
        • Prevents figure accumulation in RAM.

    Returns:
        None:
            Operates via visualization side-effects only.

    Raises:
        Exception:
            Captured internally to avoid pipeline interruption.
    """

    try:
        # Save active matplotlib figure to disk
        plt.savefig(filename)

        # Trigger Colab browser download
        files.download(filename)

        # Release figure memory resources
        plt.close()

        # Confirm successful visualization export
        print(f"[SUCCESS] Plot saved -> {filename}")

    except Exception as e:
        # Handle plotting or filesystem errors safely
        print("[ERROR] Plot saving failed:", e)

# ==========================================================
# 8A. CONFUSION MATRIX
# ==========================================================

# Notify start of confusion matrix diagnostic
print("\n[DIAGNOSTIC] Computing Confusion Matrix...")

try:

    # Recompute training predictions using LOOCV-style kNN inference
    preds_train = np.array([

        # Perform majority voting across nearest neighbors
        np.bincount(

            # Retrieve neighbor labels excluding self neighbor
            y_train[
                knn
                .kneighbors([X_train_emb[i]])[1][0][1:best_k+1]
            ],

            # Ensure all class bins exist
            minlength=10

        ).argmax()

        # Repeat prediction for every embedding sample
        for i in range(len(X_train_emb))
    ])

    # Compute confusion matrix comparing true vs predicted labels
    cm = confusion_matrix(
        y_train,
        preds_train
    )

    # Convert confusion matrix into dataframe table
    df_cm = pd.DataFrame(cm)

    # Initialize visualization canvas
    plt.figure(figsize=(8,6))

    # Plot heatmap representation of classification outcomes
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues"
    )

    # Add descriptive visualization title
    plt.title("Confusion Matrix")

    # Save and download confusion matrix figure
    save_and_download_plot(
        "confusion_matrix.png"
    )

    # Save and download confusion matrix numeric table
    download_table(
        df_cm,
        "confusion_matrix_table.csv"
    )

# Handle diagnostic computation failures safely
except Exception as e:
    print("Confusion matrix failed:", e)


# ==========================================================
# 8B. t-SNE EMBEDDING VISUALIZATION
# ==========================================================

# Notify start of nonlinear embedding visualization
print("\n[DIAGNOSTIC] Running t-SNE...")

try:

    # Initialize t-SNE dimensionality reduction model
    tsne = TSNE(
        n_components=2,      # Project embeddings into 2D space
        random_state=SEED    # Ensure reproducible projection
    )

    # Learn low-dimensional manifold representation
    X_tsne = tsne.fit_transform(
        X_train_emb          # High-dimensional ConvNeXt embeddings
    )

    # Convert projected coordinates into structured table
    df_tsne = pd.DataFrame({
        "x": X_tsne[:,0],    # First latent visualization axis
        "y": X_tsne[:,1],    # Second latent visualization axis
        "label": y_train     # Ground-truth class labels
    })

    # Create visualization canvas
    plt.figure(figsize=(8,6))

    # Plot embedding clusters per class
    for cls in range(10):

        # Scatter plot samples belonging to current class
        plt.scatter(
            df_tsne[df_tsne.label == cls]["x"],
            df_tsne[df_tsne.label == cls]["y"],
            label=str(cls),
            alpha=0.6         # Improve overlap visibility
        )

    # Display class legend
    plt.legend()

    # Add descriptive visualization title
    plt.title("t-SNE Embedding Space")

    # Save and download visualization image
    save_and_download_plot("tsne_plot.png")

    # Save projected coordinates as downloadable table
    download_table(
        df_tsne,
        "tsne_table.csv"
    )

# Handle dimensionality reduction failures safely
except Exception as e:
    print("t-SNE failed:", e)


# ==========================================================
# 8C. ROC CURVES (OVR)
# ==========================================================

# Notify start of ROC diagnostic computation
print("\n[DIAGNOSTIC] Computing ROC Curves...")

try:

    # Convert integer labels into One-vs-Rest binary format
    y_train_bin = label_binarize(
        y_train,
        classes=np.arange(10)
    )

    # Container storing ROC curve coordinates
    roc_rows = []

    # Initialize ROC visualization canvas
    plt.figure(figsize=(10,8))

    # Compute ROC curve independently for each class
    for c in range(10):

        # Compute False Positive Rate and True Positive Rate
        fpr, tpr, _ = roc_curve(
            y_train_bin[:,c],          # Binary ground truth
            preds_proba_final[:,c]     # Predicted class probability
        )

        # Compute Area Under Curve score
        roc_auc_val = auc(fpr, tpr)

        # Plot ROC curve for current class
        plt.plot(
            fpr,
            tpr,
            label=f"Class {c} AUC={roc_auc_val:.3f}"
        )

        # Store ROC coordinates for tabular export
        for fp, tp in zip(fpr, tpr):
            roc_rows.append(
                [c, fp, tp, roc_auc_val]
            )

    # Plot random-classifier reference diagonal
    plt.plot([0,1],[0,1],'--')

    # Add descriptive visualization title
    plt.title("ROC Curves")

    # Save and download ROC visualization
    save_and_download_plot(
        "roc_curves.png"
    )

    # Convert ROC coordinate log into dataframe
    df_roc = pd.DataFrame(
        roc_rows,
        columns=["class","fpr","tpr","auc"]
    )

    # Save ROC numerical results table
    download_table(
        df_roc,
        "roc_table.csv"
    )

# Handle ROC computation failures safely
except Exception as e:
    print("ROC failed:", e)


# ==========================================================
# 8D. ACCURACY VS K
# ==========================================================

# Print diagnostic header
print("\n[DIAGNOSTIC] Accuracy vs k")

try:

    # Convert stored k-results into DataFrame
    df_k = pd.DataFrame(results_k)

    # Create new figure
    plt.figure()

    # Plot accuracy against k values
    plt.plot(
        df_k["k"],
        df_k["accuracy"],
        marker="o"
    )

    # Enable grid for readability
    plt.grid(True)

    # Add plot title
    plt.title("Accuracy vs k")

    # Save and download the plot
    save_and_download_plot(
        "accuracy_vs_k.png"
    )

    # Save and download results table
    download_table(
        df_k,
        "accuracy_vs_k_table.csv"
    )

# Catch and print any errors
except Exception as e:
    print("Accuracy-k failed:", e)


# ==========================================================
# 8E. DISTANCE METRIC COMPARISON
# ==========================================================

# Print diagnostic header
print("\n[DIAGNOSTIC] Distance Metric Comparison")

# Store metric comparison results
metric_results = []

try:

    # Loop through different distance metrics
    for metric in [
        "cosine",
        "euclidean",
        "manhattan"
    ]:

        # Create KNN model using selected metric
        knn = KNeighborsClassifier(
            n_neighbors=best_k,
            metric=metric
        )

        # Train model on embeddings
        knn.fit(
            X_train_emb,
            y_train
        )

        # Find nearest neighbours
        _, indices = knn.kneighbors(
            X_train_emb
        )

        # Extract neighbour labels (excluding self neighbour)
        labels = y_train[
            indices[:, 1:best_k + 1]
        ]

        # Predict class using majority voting
        preds = np.array([
            np.bincount(
                r,
                minlength=10
            ).argmax()
            for r in labels
        ])

        # Compute accuracy
        acc = accuracy_score(
            y_train,
            preds
        )

        # Store metric performance
        metric_results.append(
            {
                "metric": metric,
                "accuracy": acc
            }
        )

    # Convert results to DataFrame
    df_metric = pd.DataFrame(
        metric_results
    )

    # Create bar chart
    plt.figure()

    # Plot accuracy for each metric
    plt.bar(
        df_metric.metric,
        df_metric.accuracy
    )

    # Add plot title
    plt.title("Accuracy vs Metric")

    # Save and download plot
    save_and_download_plot(
        "accuracy_vs_metric.png"
    )

    # Save and download results table
    download_table(
        df_metric,
        "accuracy_vs_metric_table.csv"
    )

# Catch and print any errors
except Exception as e:
    print("Metric comparison failed:", e)


# ==========================================================
# 8F. FEATURE EXTRACTION COMPARISON
# ==========================================================

# Print diagnostic header
print("\n[DIAGNOSTIC] Feature Extraction Comparison")

# Store feature extraction results
feature_results = []

try:

    # Add ConvNeXt embedding baseline accuracy
    feature_results.append(
        {
            "feature": "ConvNeXt",
            "accuracy": best_score
        }
    )

    # ------------------------------------------------------
    # PCA FEATURE EXTRACTION
    # ------------------------------------------------------

    # Initialise PCA dimensionality reduction
    pca = PCA(
        n_components=50,
        random_state=SEED
    )

    # Flatten images and apply PCA transformation
    X_pca = pca.fit_transform(
        X_train.reshape(
            len(X_train), -1
        )
    )

    # Train KNN using PCA features
    knn.fit(X_pca, y_train)

    # Retrieve nearest neighbours
    _, idx = knn.kneighbors(X_pca)

    # Extract neighbour labels
    labels = y_train[
        idx[:, 1:best_k + 1]
    ]

    # Predict labels via majority voting
    preds = np.array([
        np.bincount(
            r,
            minlength=10
        ).argmax()
        for r in labels
    ])

    # Store PCA accuracy result
    feature_results.append({
        "feature": "PCA",
        "accuracy": accuracy_score(
            y_train,
            preds
        )
    })

    # ------------------------------------------------------
    # ISOMAP FEATURE EXTRACTION
    # ------------------------------------------------------

    # Initialise Isomap dimensionality reduction
    iso = Isomap(n_components=50)

    # Flatten images and apply Isomap transformation
    X_iso = iso.fit_transform(
        X_train.reshape(
            len(X_train), -1
        )
    )

    # Train KNN using Isomap features
    knn.fit(X_iso, y_train)

    # Retrieve nearest neighbours
    _, idx = knn.kneighbors(X_iso)

    # Extract neighbour labels
    labels = y_train[
        idx[:, 1:best_k + 1]
    ]

    # Predict labels via majority voting
    preds = np.array([
        np.bincount(
            r,
            minlength=10
        ).argmax()
        for r in labels
    ])

    # Store Isomap accuracy result
    feature_results.append({
        "feature": "Isomap",
        "accuracy": accuracy_score(
            y_train,
            preds
        )
    })

    # Convert results into DataFrame
    df_fe = pd.DataFrame(
        feature_results
    )

    # Create comparison bar chart
    plt.figure()

    # Plot accuracy for each feature extraction method
    plt.bar(
        df_fe.feature,
        df_fe.accuracy
    )

    # Limit accuracy axis between 0 and 1
    plt.ylim(0, 1)

    # Add plot title
    plt.title(
        "Feature Extraction Accuracy"
    )

    # Save and download plot
    save_and_download_plot(
        "feature_accuracy.png"
    )

    # Save and download comparison table
    download_table(
        df_fe,
        "feature_accuracy_table.csv"
    )

# Catch and print errors
except Exception as e:
    print("Feature comparison failed:", e)

# ==========================================================
# 8G. ACCURACY VS PROBABILITY ESTIMATION METHOD
# ==========================================================

print("\n[DIAGNOSTIC] Comparing Probability Aggregation Methods")


def softmax_weighted_knn_predictions(
    knn_model: KNeighborsClassifier,
    embeddings: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute predictions using distance-weighted softmax voting.

    Args:
        knn_model (KNeighborsClassifier):
            Fitted kNN model.

        embeddings (np.ndarray):
            Feature embedding matrix.

        k (int):
            Number of neighbours used for voting.

    Behavior:
        • Retrieves neighbour distances
        • Converts distances → exponential weights
        • Accumulates weighted class scores
        • Performs weighted majority voting

        Unlike standard kNN averaging,
        closer neighbours contribute MORE influence.

    Returns:
        np.ndarray:
            Predicted class labels.
    """

    try:

        print("[INFO] Retrieving neighbour distances...")

        # Obtain neighbour distances + indices
        distances, indices = knn_model.kneighbors(
            embeddings
        )

        print("[INFO] Removing self-neighbour for LOOCV consistency")

        # Remove self neighbour
        neighbor_labels = y_train[
            indices[:, 1:k+1]
        ]

        # Convert distance → similarity weight
        weights = np.exp(
            -distances[:, 1:k+1]
        )

        predictions = []

        print("[INFO] Performing weighted voting...")

        # Iterate per sample
        for row_labels, row_weights in zip(
            neighbor_labels,
            weights
        ):

            # Initialize class accumulator
            class_scores = np.zeros(10)

            # Accumulate weighted votes
            for lbl, w in zip(
                row_labels,
                row_weights
            ):
                class_scores[lbl] += w

            # Select highest weighted score
            predictions.append(
                class_scores.argmax()
            )

        return np.array(predictions)

    except Exception as e:
        print("[ERROR] Softmax weighting failed:", e)
        raise



try:

    print("[INFO] Training reference kNN model...")

    knn_probability = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="cosine"
    )

    knn_probability.fit(
        X_train_emb,
        y_train
    )

    # ======================================================
    # STANDARD AVERAGE VOTING
    # ======================================================

    print("[INFO] Computing standard average-vote accuracy...")

    preds_avg = np.array([
        np.bincount(
            y_train[
                knn_probability
                .kneighbors([X_train_emb[i]])[1][0][1:best_k+1]
            ],
            minlength=10
        ).argmax()
        for i in range(len(X_train_emb))
    ])

    acc_avg = accuracy_score(
        y_train,
        preds_avg
    )

    # ======================================================
    # SOFTMAX WEIGHTED VOTING
    # ======================================================

    print("[INFO] Computing softmax-weighted predictions...")

    preds_softmax = softmax_weighted_knn_predictions(
        knn_probability,
        X_train_emb,
        best_k
    )

    acc_softmax = accuracy_score(
        y_train,
        preds_softmax
    )

    print(
        f"[RESULT] Avg Accuracy={acc_avg:.4f} "
        f"| Softmax Accuracy={acc_softmax:.4f}"
    )

    # ======================================================
    # RESULT TABLE
    # ======================================================

    df_probability = pd.DataFrame([
        {
            "method": "average_vote",
            "accuracy": acc_avg
        },
        {
            "method": "softmax_weighted",
            "accuracy": acc_softmax
        }
    ])

    # ======================================================
    # VISUALIZATION
    # ======================================================

    plt.figure()

    plt.bar(
        df_probability["method"],
        df_probability["accuracy"]
    )

    plt.ylabel("Accuracy")

    plt.title(
        "Accuracy vs Probability Method"
    )

    save_and_download_plot(
        "accuracy_vs_probability.png"
    )

    download_table(
        df_probability,
        "accuracy_vs_probability_table.csv"
    )

except Exception as e:
    print(
        "[ERROR] Probability comparison failed:",
        e
    )

# ==========================================================
# 9. FINAL EXPORTS
# ==========================================================

# Display export stage header
print("\n[EXPORT] Saving predictions...")

try:

    # ------------------------------------------------------
    # EXPORT MODEL PREDICTIONS
    # ------------------------------------------------------

    # Create dataframe containing test labels
    df_knn_predictions = pd.DataFrame({
        "label": preds
    })

    # Save predictions locally
    df_knn_predictions.to_csv(
        "knn_predictions.csv",
        index=False
    )

    # ------------------------------------------------------
    # EXPORT WEBSITE DEPLOYMENT OUTPUT
    # ------------------------------------------------------

    # Create dataframe with probabilities
    df_website_predictions = pd.DataFrame({
        "label": preds,
        "probability": preds_proba_final.max(axis=1)
    })

    # Save deployment-ready predictions
    df_website_predictions.to_csv(
        "website_predictions.csv",
        index=False
    )

    # ------------------------------------------------------
    # TRIGGER DOWNLOADS
    # ------------------------------------------------------

    # Download prediction file
    files.download(
        "knn_predictions.csv"
    )

    # Download website prediction file
    files.download(
        "website_predictions.csv"
    )

    # Confirm successful export
    print("✅ Prediction exports completed")

# Handle export failure
except Exception as e:
    print("Export failed:", e)


# Display final pipeline completion message
print(
    "\n✅✅✅ FULL PIPELINE EXECUTED SUCCESSFULLY"
)