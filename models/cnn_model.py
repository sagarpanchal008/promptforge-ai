"""
==============================================================================
FILE: models/cnn_model.py
PURPOSE:
    Defines the 1D Convolutional Neural Network (1D-CNN) architecture for
    prompt quality classification (Poor, Average, Good).
    Extracts local n-gram semantic patterns from word embedding sequences.

ARCHITECTURE OVERVIEW:
    1. Input Layer: Padded sequence of token indices (shape: [batch_size, max_len=120])
    2. Embedding Layer: Maps discrete token IDs to dense continuous vectors (dim=128)
    3. Conv1D Layer 1: 128 filters, kernel_size=5 (captures 5-word local phrases / n-grams)
    4. ReLU Activation: Introduces non-linearity max(0, z)
    5. MaxPooling1D Layer: Pool size 2 (halves temporal dimension, retains dominant activations)
    6. Conv1D Layer 2: 64 filters, kernel_size=3 (captures 3-word hierarchical patterns)
    7. GlobalMaxPooling1D Layer: Extracts the most salient feature per filter map across the entire prompt
    8. Dense Hidden Layer: 64 units with ReLU activation (high-level feature combination)
    9. Dropout Layer: Rate 0.4 (regularization to prevent co-adaptation / overfitting)
    10. Output Dense Layer: 3 units with Softmax activation (outputs class probability distribution)

ACADEMIC EXPLANATION FOR VIVA:
    Why 1D-CNN for Text?
    While CNNs are famously used for 2D images, 1D-CNNs operate over text by sliding
    convolutional filters along the temporal sequence of word embeddings. A filter of
    size 5 acts like a learnable 5-gram detector, recognizing key prompt cues like
    "Act as an expert", "write a function", or "give me code" regardless of where
    they appear in the prompt.
==============================================================================
"""

from typing import Tuple


def build_cnn_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_len: int = 120,
    num_classes: int = 3,
    dropout_rate: float = 0.4,
) -> "keras.Model":
    """
    Constructs and compiles the 1D-CNN prompt quality classifier.

    Args:
        vocab_size: Total vocabulary size (including <PAD> and <UNK>).
        embedding_dim: Dimensionality of dense word embedding vectors.
        max_len: Fixed sequence length of token inputs.
        num_classes: Number of target quality classes (3: Poor, Average, Good).
        dropout_rate: Fraction of dense neurons dropped during training.

    Returns:
        Compiled TensorFlow Keras Model ready for training.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    # 1. Input Layer: Integer token IDs of fixed length (120)
    inputs = layers.Input(shape=(max_len,), name="token_input", dtype="int32")

    # 2. Embedding Layer: Converts token indices [0..vocab_size] into 128-dim dense vectors
    # mask_zero=False because Conv1D does not support standard zero-masking
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="embedding_layer",
    )(inputs)

    # 3. First 1D Convolution: 128 filters of window length 5 (detects 5-gram patterns)
    conv1 = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1d_layer_1",
    )(embedding)

    # 4. Max Pooling: Downsamples sequence by factor of 2, preserving highest activation
    pool1 = layers.MaxPooling1D(pool_size=2, name="maxpool1d_1")(conv1)

    # 5. Second 1D Convolution: 64 filters of window length 3 (detects 3-gram patterns)
    conv2 = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1d_layer_2",
    )(pool1)

    # 6. Global Max Pooling: Condenses temporal dimension into a fixed-size 64-dim vector
    # capturing the strongest signal from each filter across the entire prompt
    global_pool = layers.GlobalMaxPooling1D(name="global_max_pooling")(conv2)

    # 7. Fully Connected Dense Layer: High-level classification reasoning
    dense = layers.Dense(64, activation="relu", name="dense_features")(global_pool)

    # 8. Dropout Layer: Regularization to combat overfitting
    dropout = layers.Dropout(dropout_rate, name="dropout_regularization")(dense)

    # 9. Output Layer: Probability distribution over 3 classes (Poor, Average, Good)
    outputs = layers.Dense(num_classes, activation="softmax", name="quality_probabilities")(dropout)

    model = models.Model(inputs=inputs, outputs=outputs, name="Prompt_Quality_1D_CNN")

    # Compile with Adam optimizer and Categorical Crossentropy Loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    try:
        import tensorflow as tf
        model = build_cnn_model(vocab_size=1343, max_len=120, num_classes=3)
        model.summary()
        print("\n[+] 1D-CNN Model Architecture built and validated successfully!")
    except ImportError:
        print("[!] TensorFlow not installed in local environment.")
        print("[*] Architecture definition is ready for Google Colab execution.")
