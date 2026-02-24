"""
====================================================================
Plant Disease Detection - Model Training Script
====================================================================
Author      : Senior AI Engineer
Description : Transfer Learning with MobileNetV2 on PlantVillage
Dataset     : PlantVillage (38 classes)
Architecture: MobileNetV2 + Custom Dense Layers
====================================================================

HOW TO RETRAIN:
1. Download PlantVillage dataset from Kaggle:
   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Extract to: ./data/PlantVillage/
3. Run: python model/train.py
4. Model saved to: model/plant_model.h5

FOLDER STRUCTURE EXPECTED:
data/
└── PlantVillage/
    ├── Tomato_Bacterial_spot/
    ├── Tomato_Early_blight/
    ├── Tomato_healthy/
    └── ... (38 class folders)
====================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"       : "./data/PlantVillage",
    "model_save_path": "./model/plant_model.h5",
    "labels_path"    : "./model/class_labels.json",
    "img_size"       : (224, 224),
    "batch_size"     : 32,
    "epochs"         : 30,
    "learning_rate"  : 1e-4,
    "val_split"      : 0.2,
    "seed"           : 42,
    "fine_tune_at"   : 100,   # Unfreeze MobileNetV2 layers from this index
    "dropout_rate"   : 0.3,
}

print("=" * 60)
print("  Plant Disease Detection - Model Training")
print(f"  TensorFlow Version: {tf.__version__}")
print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: DATA AUGMENTATION & GENERATORS
# ─────────────────────────────────────────────
def create_data_generators(config):
    """
    Create train and validation data generators with augmentation.
    
    Why augmentation?
    - Prevents overfitting on limited data
    - Makes model robust to real-world variations
    - Simulates different lighting, angles, conditions
    """
    print("\n[1/6] Setting up data generators...")

    # Training augmentation (aggressive for better generalization)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              # Normalize pixel values [0,1]
        validation_split=config["val_split"],
        rotation_range=30,              # Random rotation up to 30°
        width_shift_range=0.15,         # Horizontal shift
        height_shift_range=0.15,        # Vertical shift
        shear_range=0.1,                # Shear transformation
        zoom_range=0.2,                 # Random zoom
        horizontal_flip=True,           # Mirror flip
        vertical_flip=False,            # Don't flip vertically (leaves)
        brightness_range=[0.8, 1.2],    # Vary brightness
        fill_mode='nearest'             # Fill empty pixels
    )

    # Validation: only rescale (no augmentation!)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config["val_split"]
    )

    train_gen = train_datagen.flow_from_directory(
        config["data_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode='categorical',
        subset='training',
        seed=config["seed"],
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        config["data_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode='categorical',
        subset='validation',
        seed=config["seed"],
        shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    class_labels = {v: k for k, v in train_gen.class_indices.items()}

    print(f"  ✓ Training samples  : {train_gen.samples}")
    print(f"  ✓ Validation samples: {val_gen.samples}")
    print(f"  ✓ Number of classes : {num_classes}")

    return train_gen, val_gen, num_classes, class_labels


# ─────────────────────────────────────────────
# STEP 2: BUILD MODEL WITH TRANSFER LEARNING
# ─────────────────────────────────────────────
def build_model(num_classes, config):
    """
    Build MobileNetV2-based transfer learning model.
    
    Architecture:
    ┌──────────────────────────────────┐
    │  Input (224 x 224 x 3)          │
    ├──────────────────────────────────┤
    │  MobileNetV2 (ImageNet weights)  │ ← Frozen initially
    │  (Feature Extractor)            │
    ├──────────────────────────────────┤
    │  Global Average Pooling          │
    ├──────────────────────────────────┤
    │  Dense(512, ReLU) + BatchNorm    │
    │  Dropout(0.3)                    │
    ├──────────────────────────────────┤
    │  Dense(256, ReLU) + BatchNorm    │
    │  Dropout(0.3)                    │
    ├──────────────────────────────────┤
    │  Dense(num_classes, Softmax)     │ ← Output
    └──────────────────────────────────┘
    
    Why MobileNetV2?
    - Lightweight & fast (ideal for deployment)
    - Pre-trained on ImageNet (1M+ images)
    - Depthwise separable convolutions
    - Excellent accuracy/speed trade-off
    """
    print("\n[2/6] Building model architecture...")

    # Load MobileNetV2 without top classification layers
    base_model = MobileNetV2(
        input_shape=(*config["img_size"], 3),
        include_top=False,              # Remove ImageNet classification head
        weights='imagenet'              # Use pre-trained ImageNet weights
    )

    # Freeze all base model layers initially
    base_model.trainable = False

    # Build custom classification head
    inputs  = tf.keras.Input(shape=(*config["img_size"], 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)

    # Dense Block 1
    x       = layers.Dense(512)(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation('relu')(x)
    x       = layers.Dropout(config["dropout_rate"])(x)

    # Dense Block 2
    x       = layers.Dense(256)(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation('relu')(x)
    x       = layers.Dropout(config["dropout_rate"])(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile with Adam optimizer
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    print(f"  ✓ Base model    : MobileNetV2 ({base_model.count_params():,} params)")
    print(f"  ✓ Total params  : {model.count_params():,}")
    print(f"  ✓ Trainable     : {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    return model, base_model


# ─────────────────────────────────────────────
# STEP 3: CALLBACKS
# ─────────────────────────────────────────────
def create_callbacks(config):
    """Set up training callbacks for best performance."""
    print("\n[3/6] Configuring callbacks...")

    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    callbacks = [
        # Save best model based on val_accuracy
        ModelCheckpoint(
            filepath=config["model_save_path"],
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),

        # Stop training if no improvement for 8 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            histogram_freq=1
        )
    ]

    print("  ✓ ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard")
    return callbacks


# ─────────────────────────────────────────────
# STEP 4: PHASE 1 - FEATURE EXTRACTION
# ─────────────────────────────────────────────
def train_phase1(model, train_gen, val_gen, callbacks, config):
    """
    Phase 1: Train only the custom classification head.
    Base MobileNetV2 is frozen - we only learn how to classify.
    """
    print("\n[4/6] Phase 1: Training classification head...")
    print("      (MobileNetV2 frozen - feature extraction only)")

    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  ✓ Phase 1 complete.")
    print(f"  Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")
    return history1


# ─────────────────────────────────────────────
# STEP 4b: PHASE 2 - FINE TUNING
# ─────────────────────────────────────────────
def train_phase2(model, base_model, train_gen, val_gen, config):
    """
    Phase 2: Fine-tune upper layers of MobileNetV2.
    Unfreeze last N layers and train with very low LR.
    """
    print("\n      Phase 2: Fine-tuning MobileNetV2 upper layers...")
    print(f"      (Unfreezing from layer {config['fine_tune_at']})")

    # Unfreeze base model partially
    base_model.trainable = True
    for layer in base_model.layers[:config["fine_tune_at"]]:
        layer.trainable = False

    # Recompile with much lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    fine_tune_callbacks = [
        ModelCheckpoint(
            filepath=config["model_save_path"],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8)
    ]

    history2 = model.fit(
        train_gen,
        epochs=config["epochs"],
        validation_data=val_gen,
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    print(f"  ✓ Phase 2 complete.")
    print(f"  Best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    return history2


# ─────────────────────────────────────────────
# STEP 5: VISUALIZATION & EVALUATION
# ─────────────────────────────────────────────
def plot_training_history(history1, history2=None):
    """Plot accuracy and loss curves."""
    print("\n[5/6] Generating training plots...")

    # Combine both phases if available
    if history2:
        acc  = history1.history['accuracy'] + history2.history['accuracy']
        val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        vloss= history1.history['val_loss'] + history2.history['val_loss']
        split_epoch = len(history1.history['accuracy'])
    else:
        acc   = history1.history['accuracy']
        val   = history1.history['val_accuracy']
        loss  = history1.history['loss']
        vloss = history1.history['val_loss']
        split_epoch = None

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Plant Disease Detection - Training Results', fontsize=16, fontweight='bold')

    # Accuracy plot
    axes[0].plot(epochs, acc, 'b-o', label='Training Accuracy', markersize=3)
    axes[0].plot(epochs, val, 'r-o', label='Validation Accuracy', markersize=3)
    if split_epoch:
        axes[0].axvline(x=split_epoch, color='g', linestyle='--', label='Fine-tuning starts')
    axes[0].set_title('Model Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss plot
    axes[1].plot(epochs, loss,  'b-o', label='Training Loss',   markersize=3)
    axes[1].plot(epochs, vloss, 'r-o', label='Validation Loss',  markersize=3)
    if split_epoch:
        axes[1].axvline(x=split_epoch, color='g', linestyle='--', label='Fine-tuning starts')
    axes[1].set_title('Model Loss', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./model/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Training plot saved: model/training_history.png")


def evaluate_model(model, val_gen, class_labels):
    """Generate confusion matrix and classification report."""
    print("\n[6/6] Evaluating model...")

    # Collect predictions
    val_gen.reset()
    y_true, y_pred = [], []

    for i in range(len(val_gen)):
        imgs, labels = val_gen[i]
        preds        = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Class names
    class_names = [class_labels[i] for i in range(len(class_labels))]

    # --- Classification Report ---
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n  Classification Report:\n")
    print(report)

    with open('./model/classification_report.txt', 'w') as f:
        f.write(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.title('Confusion Matrix - Plant Disease Detection', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./model/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  ✓ Confusion matrix saved: model/confusion_matrix.png")
    print("  ✓ Report saved: model/classification_report.txt")

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\n  ✓ Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    # Set seeds for reproducibility
    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])

    # Check dataset
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"\n  ❌ Dataset not found at: {CONFIG['data_dir']}")
        print("  Please download PlantVillage from Kaggle:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        return

    # Run pipeline
    train_gen, val_gen, num_classes, class_labels = create_data_generators(CONFIG)

    # Save class labels
    with open(CONFIG["labels_path"], 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"  ✓ Class labels saved: {CONFIG['labels_path']}")

    model, base_model = build_model(num_classes, CONFIG)
    callbacks = create_callbacks(CONFIG)

    # Phase 1: Feature extraction
    history1 = train_phase1(model, train_gen, val_gen, callbacks, CONFIG)

    # Phase 2: Fine-tuning
    history2 = train_phase2(model, base_model, train_gen, val_gen, CONFIG)

    # Visualize
    plot_training_history(history1, history2)

    # Evaluate
    evaluate_model(model, val_gen, class_labels)

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"  Model saved: {CONFIG['model_save_path']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
