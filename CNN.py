import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# Pasul 1: Inițializarea
np.random.seed(42)
data_root = 'D:\\Master 2\\TIDAIM\\archive\\Data\\train'
label_map = {
    'adenocarcinoma': 0,
    'large.cell.carcinoma': 1,
    'normal': 2,
    'squamous.cell.carcinoma': 3
}

# Pasul 2: Încărcarea datelor
def load_data(data_dir, label_map):
    images = []
    labels = []
    for folder_name, label in label_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path).convert('L')  # Converteste la scala de gri
            image = image.resize((128, 128))  # Redimensionare
            images.append(np.array(image))
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_data(data_root, label_map)

# Pasul 3: Verificarea dimensiunii datelor
print("Dimensiunea imaginilor: ", images.shape)
print("Dimensiunea etichetelor: ", labels.shape)

# Pasul 3: Preprocesarea datelor
# Normalizarea pixelilor
images = images / 255.0
# Convertirea etichetelor în categorii
labels = to_categorical(labels, num_classes=len(label_map))

# Data Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Divizarea datelor în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 128, 128, 1)
# Define the CNN model with Batch Normalization and L1/L2 Regularization
model = Sequential()

# Convolutional blocks
model.add(Conv2D(32, (3, 3), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Adăugarea mai multor straturi Convolutional și MaxPooling
model.add(Conv2D(64, (3, 3), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Al treilea set de straturi Convolutional + MaxPooling
model.add(Conv2D(128, (3, 3), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Al patrulea set de straturi Convolutional + MaxPooling
model.add(Conv2D(256, (3, 3), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Strat de aplatizare
model.add(Flatten())

# Strat dens
# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.5))  # You can experiment with this rate
model.add(Dense(len(label_map), activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Pasul 5: Compilarea modelului
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Pasul 6: Antrenarea modelului
# history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
history = model.fit(
datagen.flow(X_train, y_train, batch_size=32),
steps_per_epoch=len(X_train) / 32, # Number of steps per epoch
epochs=20, # Start with a high number, early stopping will stop the training
validation_data=(X_test, y_test),
callbacks=[reduce_lr, early_stop]
)
# Pasul 7: Evaluarea modelului
# loss, accuracy = model.evaluate(X_test, y_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


    