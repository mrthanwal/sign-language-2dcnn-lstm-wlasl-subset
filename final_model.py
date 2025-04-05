import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import os

# Load the data
def load_data(split_folder, max_frames=15, num_features=126):
    x_data = np.load(os.path.join(split_folder, "x_train.npz"))
    y_data = np.load(os.path.join(split_folder, "y_train.npz"))

    X_list = []
    Y_list = []

    for key in x_data.files:
        sequence = x_data[key]  

        # Ensure uniform shape (pad or trim)
        if sequence.shape[0] < max_frames:
            pad_width = ((0, max_frames - sequence.shape[0]), (0, 0))
            sequence = np.pad(sequence, pad_width, mode='constant')
        elif sequence.shape[0] > max_frames:
            sequence = sequence[:max_frames, :]

        X_list.append(sequence)
        Y_list.append(y_data[key])  

    X = np.array(X_list)  
    Y = np.array(Y_list)  

    return X, Y

split_folder = "/kaggle/input/data-lauda"

X_train, Y_train = load_data(split_folder)
X_val, Y_val = load_data(split_folder)
X_test, Y_test = load_data(split_folder)

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)  
Y_val = label_encoder.transform(Y_val)
Y_test = label_encoder.transform(Y_test)

num_classes = len(label_encoder.classes_)  
Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)
Y_test = to_categorical(Y_test, num_classes)

print("Class Mapping:", dict(zip(label_encoder.classes_, range(num_classes))))

X_train = X_train.reshape(-1, 15, 126, 1)
X_val = X_val.reshape(-1, 15, 126, 1)
X_test = X_test.reshape(-1, 15, 126, 1)

input_shape = (15, 126, 1)

def compute_class_weights(y_train, num_classes):
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    return dict(enumerate(class_weights))

image_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fixed Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1, 2)),

    # Corrected Flattening and Reshaping
    Flatten(),
    Reshape((15, -1)),  # Automatically calculates the correct shape

    # Corrected LSTM Layers
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0003),
              loss='categorical_crossentropy',  # Corrected loss function
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=8
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Save the model
model.save("sign_language_model.h5")
