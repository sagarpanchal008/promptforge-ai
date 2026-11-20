"""
==============================================================================
FILE: models/lstm_model.py
PURPOSE:
    Defines the Bidirectional Long Short-Term Memory (Bi-LSTM) architecture
    for prompt quality assessment and sequential context modeling.
    Unlike CNNs which focus on local n-grams, Bi-LSTM analyzes long-range
    temporal dependencies, sentence structure, and context flow.

ARCHITECTURE OVERVIEW:
    1. Input Layer: Padded integer token sequences (shape: [batch_size, max_len=120])
    2. Embedding Layer: Converts token IDs into dense 128-dimensional word vectors
    3. SpatialDropout1D: Drops 1D feature channels (rate=0.2) to prevent overfitting
    4. Bidirectional LSTM Layer 1: 64 units per direction (128 total), return_sequences=True
       - Forward LSTM: Reads the prompt from beginning to end
       - Backward LSTM: Reads the prompt from end to beginning
    5. Bidirectional LSTM Layer 2: 32 units per direction (64 total), return_sequences=False
       - Condenses temporal information into a single context summary vector
    6. Dense Layer: 64 hidden units with ReLU activation and L2 weight regularization
    7. Dropout Layer: Regularization rate 0.4
    8. Dense Output Layer: 3 units with Softmax activation (Poor, Average, Good)

ACADEMIC COMPARISON FOR VIVA (CNN vs Bi-LSTM):
    - CNN: Uses fixed-size sliding kernels (local n-grams) to spot specific keywords.
      Very fast, but ignores long-distance syntactic dependencies.
    - Bi-LSTM: Maintains an internal memory cell (c_t) and hidden state (h_t) controlled
      by Input, Forget, and Output gates. It remembers whether instructions started with
      a role and ended with specific output constraints across 100+ tokens.
==============================================================================
"""

from typing import Tuple


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_len: int = 120,
    num_classes: int = 3,
    dropout_rate: float = 0.4,
) -> "keras.Model":
    """
    Constructs and compiles the Bidirectional LSTM prompt quality model.

    Args:
        vocab_size: Size of vocabulary including <PAD> and <UNK>.
        embedding_dim: Dense word vector dimension.
        max_len: Uniform sequence token length (120).
        num_classes: 3 target classes (Poor, Average, Good).
        dropout_rate: Dense dropout fraction.

    Returns:
        Compiled Keras Model instance ready for training.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    # 1. Input Layer: Token sequence IDs
    inputs = layers.Input(shape=(max_len,), name="token_input", dtype="int32")

    # 2. Embedding Layer: Continuous semantic word representations
    # mask_zero=True allows LSTM to dynamically skip padding tokens (zeros)
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True,
        name="embedding_layer",
    )(inputs)

    # 3. Spatial Dropout: Drops entire 1D word feature maps
    spatial_dropout = layers.SpatialDropout1D(0.2, name="spatial_dropout")(embedding)

    # 4. First Bi-LSTM: Captures bidirectional sequential patterns (64 forward + 64 backward = 128)
    bilstm1 = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            name="forward_backward_lstm_1",
        ),
        name="bidirectional_lstm_1",
    )(spatial_dropout)

    # 5. Second Bi-LSTM: Compresses full sequence into context vector (32 forward + 32 backward = 64)
    bilstm2 = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=False,
            dropout=0.2,
            name="forward_backward_lstm_2",
        ),
        name="bidirectional_lstm_2",
    )(bilstm1)

    # 6. Dense Hidden Layer: Non-linear reasoning combining forward and backward context
    dense = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="context_dense",
    )(bilstm2)

    # 7. Dense Dropout Layer: Regularization against co-adaptation
    dropout = layers.Dropout(dropout_rate, name="dense_dropout")(dense)

    # 8. Output Layer: Softmax probability distribution over 3 quality classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="quality_probabilities",
    )(dropout)

    model = models.Model(inputs=inputs, outputs=outputs, name="Prompt_Quality_BiLSTM")

    # Compile with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    try:
        import tensorflow as tf
        model = build_lstm_model(vocab_size=1343, max_len=120, num_classes=3)
        model.summary()
        print("\n[+] Bidirectional LSTM Model Architecture built and validated successfully!")
    except ImportError:
        print("[!] TensorFlow not installed in current environment.")
