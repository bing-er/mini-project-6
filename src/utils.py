import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.efficientnet import decode_predictions as effnet_decode
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def get_layer_shape(layer):
    """Safely get output shape (Keras 3.x compatible)."""
    try:
        return str(layer.output.shape)
    except (AttributeError, ValueError):
        return "N/A"

def model_summary_stats(model, name):
    """Print key statistics about a model."""
    total_params = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = total_params - trainable
    num_layers = len(model.layers)
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Total layers:        {num_layers}")
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable params:    {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    print(f"Model size (approx): {total_params * 4 / (1024**2):.1f} MB (float32)")
    print(f"Input shape:         {model.input_shape}")
    print(f"Output shape:        {model.output_shape}")
    return {'name': name, 'layers': num_layers, 'params': total_params}

def show_trainable_stats(model):
    """Show trainable vs frozen parameter counts."""
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total = trainable + non_trainable
    print(f"\n{model.name}")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,} ({trainable/total:.1%})")
    print(f"  Frozen parameters:    {non_trainable:>12,} ({non_trainable/total:.1%})")

def plot_confusion_matrix(model, dataset, class_names, title):
    """Generate and plot confusion matrix."""
    y_true, y_pred = [], []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=13)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    print(f"\n{title} — Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def class_counts(ds):
    counts = Counter()
    for batch in ds:
        # labeled: (images, labels)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            _, labels = batch

            # labels can be int or one-hot
            if len(labels.shape) > 1:
                labels = tf.argmax(labels, axis=1)

            counts.update(labels.numpy().tolist())

        else:
            # unlabeled: only images
            raise ValueError("Dataset has no labels, cannot compute class counts.")
    return counts

def build_feature_extraction_model(base_model_class, preprocess_fn, name, data_augmentation, num_classes=5):
    """Build a feature extraction model using a pre-trained backbone."""

    # Step 1: Load pre-trained model WITHOUT the classifier
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,          
        input_shape=(224, 224, 3)
    )

    # Step 2: FREEZE all pre-trained layers
    # TODO: Set base_model.trainable to False
    base_model.trainable = False


    # Step 3: Build the complete model
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)       # Augmentation (only active during training)
    x = preprocess_fn(x)               # Model-specific normalization
    x = base_model(x, training=False)  # Pass through frozen backbone

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name=name)
    return model, base_model

def plot_training_history(history):
    """Plot training & validation accuracy and loss."""
    import matplotlib.pyplot as plt
    
    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="Train Accuracy")
    if val_acc:
        plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.show()