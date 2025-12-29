import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

dataset_url = "suchintikasarkar/sentiment-analysis-for-mental-health"
file_name = "Combined Data.csv"
path = kagglehub.dataset_download(dataset_url)
file_path = os.path.join(path, file_name)

print("Path to dataset files:", path)
print(f"Attempting to load data from: {file_path}")

# --- 1. Data Loading ---

df = pd.read_csv(file_path)

print("\n--- 1. DataFrame Head (First 5 Rows) ---")
print(df.head())

print("\n--- 2. DataFrame Information (Columns, Non-null counts, Dtypes) ---")
print(df.info())

print("\n--- 3. Dataset Shape (Rows, Columns) ---")
print(df.shape)

print("\n--- 4. Target Variable Distribution (Sentiment/Label) ---")
print("\nObject (Text) Column Unique Values Check:")
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < 20:
        print(f"Column '{col}' value counts:\n{df[col].value_counts()}\n")

print("\n--- 5. Check for Missing Values ---")
print(df.isnull().sum())

# --- 2. Data Cleaning and Preprocessing ---

# Handle missing values by dropping rows where 'statement' is NaN
# Since only 362 out of 53043 are missing, dropping them is generally safe.
df.dropna(subset=['statement'], inplace=True)

# Drop the irrelevant index column
df = df.drop(columns=['Unnamed: 0'])

# Drop the duplicate rows
df = df.drop_duplicates()

# Convert all text to string
df['statement'] = df['statement'].astype(str)

# --- 3. Label Encoding for the Target Variable (status) ---

# Initialize LabelEncoder
le = LabelEncoder()
# Fit and transform the 'status' column
df['status_encoded'] = le.fit_transform(df['status'])

# Display the mapping
print("\n--- Label Encoding Mapping ---")
for i, label in enumerate(le.classes_):
    print(f"{label}: {i}")

# --- 5. Data Splitting ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'],
    df['status_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['status_encoded']
)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"\nTraining on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")

# --- 6. Text Vectorization ---

# define parameters
MAX_VOCAB_SIZE = 20000  # max size of vocabulary list
MAX_SEQUENCE_LENGTH = 128  # max sequence length
EMBEDDING_DIM = 64  # embedding dimension

# create TextVectorization layer
vectorize_layer = layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    standardize='lower_and_strip_punctuation'
)

# Computes a vocabulary of string terms from tokens in a dataset
vectorize_layer.adapt(X_train)

print("The vocabulary list is complete. The first 5 words:", vectorize_layer.get_vocabulary()[:5])

# --- 7. TensorFlow Model Building ---

model = models.Sequential([
    # Input layer: Receives raw strings
    tf.keras.Input(shape=(1,), dtype=tf.string),

    # Vectorization layer: Converts strings into sequences of integers.
    vectorize_layer,

    # Embedding layer: learning word vectors
    layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),

    # Dropout layer: preventing overfitting
    layers.Dropout(0.5),

    # Bidirectional layer: capturing context information
    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),

    # Fully connected layer
    layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),

    # Dropout layer: preventing overfitting
    layers.Dropout(0.5),

    # Output layer: Multi-class Softmax
    layers.Dense(7, activation='softmax')
])

# compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --- 8. Model Training ---

# Configure Early Stopping to prevent overfitting.
'''early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)'''

EPOCHS = 4
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# --- 9. Model Evaluation ---

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# --- 10. Training Process Visualization ---
def plot_history(__history):
    acc = __history.history['accuracy']
    val_acc = __history.history['val_accuracy']
    loss = __history.history['loss']
    val_loss = __history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history)
