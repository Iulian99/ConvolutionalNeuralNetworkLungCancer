import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mlp

def load_data(data_dir, label_map):
    images = []
    labels = []

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path).convert('L')  # conversia la scala de gri
                image = image.resize((128, 128))  # redimensionare
                images.append(np.array(image) / 255.0)  # normalizare
                labels.append(label_map[folder_name])  # adăugarea etichetei

    return np.array(images), np.array(labels)

#data set & labels
data_root = 'D:/Master 2/TIDAIM/archive/Data'
label_map = {
    'adenocarcinoma': 0,
    'large.cell.carcinoma': 1,
    'normal': 2,
    'squamous.cell.carcinoma': 3

train_images, train_labels = load_data(os.path.join(data_root, 'train'), label_map)
test_images, test_labels = load_data(os.path.join(data_root, 'test'), label_map)
valid_images, valid_labels = load_data(os.path.join(data_root, 'valid'), label_map)

# Resize Images
train_images = train_images.reshape(train_images.shape[0], -1)
valid_images = valid_images.reshape(valid_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
    
# Convert Labels to One-Hot Format
def transform_to_onehot(labels, num_classes=4):
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels

#train model
train_labels_onehot = transform_to_onehot(train_labels, num_classes=4)
valid_labels_onehot = transform_to_onehot(valid_labels, num_classes=4)
test_labels_onehot = transform_to_onehot(test_labels, num_classes=4)

# First hidden layer
input_size = 128 * 128
num_classes = 4         
hidden_layer_size = 128  

# Second hidden layer
second_hidden_layer_size = 128 

w1_2 = np.random.normal(0, 0.001, (second_hidden_layer_size, hidden_layer_size))
b1_2 = np.zeros(second_hidden_layer_size)

# Third hidden layer
third_hidden_layer_size = 128

w1_3 = np.random.normal(0, 0.001, (third_hidden_layer_size, second_hidden_layer_size))
b1_3 = np.zeros(third_hidden_layer_size)

# Initialize weights and biases
w1 = np.random.normal(0, 0.001, (hidden_layer_size, input_size)) # Ponderi pentru stratul ascuns
b1 = np.zeros(hidden_layer_size)                                # Biasuri pentru stratul ascuns

# w2 = np.random.normal(0, 0.001, (num_classes, hidden_layer_size)) # Ponderi pentru stratul de ieșire

# Changing the weights for the output layer to connect to the third hidden layer
w2 = np.random.normal(0, 0.001, (num_classes, third_hidden_layer_size))
b2 = np.zeros(num_classes)                                        # Biasuri pentru stratul de ieșire


# Activation and forward propagation functions
# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilizează exponențiala
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# for 2 hidden layers
# def forward(x, w1, b1, w2, b2):
#     # Stratul ascuns
#     z1 = np.dot(x, w1.T) + b1
#     a1 = relu(z1)
#     # Stratul de ieșire
#     z2 = np.dot(a1, w2.T) + b2
#     a2 = softmax(z2)
#     return a2

# def forward(x, w1, b1, w1_2, b1_2, w2, b2):
#     z1 = np.dot(x, w1.T) + b1
#     a1 = relu(z1)
#     z1_2 = np.dot(a1, w1_2.T) + b1_2
#     a1_2 = relu(z1_2)
#     z2 = np.dot(a1_2, w2.T) + b2
#     a2 = softmax(z2)
#     return a2

# Forward propagions
def forward(x, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2):
    z1 = np.dot(x, w1.T) + b1
    a1 = relu(z1)
    z1_2 = np.dot(a1, w1_2.T) + b1_2
    a1_2 = relu(z1_2)
    z1_3 = np.dot(a1_2, w1_3.T) + b1_3
    a1_3 = relu(z1_3)
    z2 = np.dot(a1_3, w2.T) + b2
    a2 = softmax(z2)
    return a2

# Loss entropy - difference between the network's predictions and the true values(the labels).
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m 
    return loss

# y_pred = forward(train_images, w1, b1, w2, b2)
y_pred = forward(train_images, w1, b1, w1_2, b1_2,w1_3, b1_3, w2, b2)

loss = cross_entropy_loss(y_pred, train_labels_onehot)
print("Pierderea de antrenament:", loss)

# Calculation of Derivatives of Activation Functions
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Implementation for 2 hidden layers
# def backpropagation(x, y_true, w1, b1, w2, b2):
#     # Forward pass
#     z1 = np.dot(x, w1.T) + b1
#     a1 = relu(z1)
#     z2 = np.dot(a1, w2.T) + b2
#     a2 = softmax(z2)

#     # Calculul erorii la stratul de ieșire
#     error_output = a2 - y_true

#     # ponderile și biasurile stratului de ieșire
#     dw2 = np.dot(error_output.T, a1) / x.shape[0]
#     db2 = np.sum(error_output, axis=0) / x.shape[0]

#     # Propagarea erorii înapoi la stratul ascuns
#     error_hidden = np.dot(error_output, w2) * relu_derivative(z1)

#     # Calculul gradienților pentru ponderile și biasurile stratului ascuns
#     dw1 = np.dot(error_hidden.T, x) / x.shape[0]
#     db1 = np.sum(error_hidden, axis=0) / x.shape[0]

#     return dw1, db1, dw2, db2

# def backpropagation(x, y_true, w1, b1, w1_2, b1_2, w2, b2):
#     z1 = np.dot(x, w1.T) + b1
#     a1 = relu(z1)
#     z1_2 = np.dot(a1, w1_2.T) + b1_2
#     a1_2 = relu(z1_2)
#     z2 = np.dot(a1_2, w2.T) + b2
#     a2 = softmax(z2)

#     error_output = a2 - y_true
#     dw2 = np.dot(error_output.T, a1_2) / x.shape[0]
#     db2 = np.sum(error_output, axis=0) / x.shape[0]

#     error_hidden_2 = np.dot(error_output, w2) * relu_derivative(z1_2)
#     dw1_2 = np.dot(error_hidden_2.T, a1) / x.shape[0]
#     db1_2 = np.sum(error_hidden_2, axis=0) / x.shape[0]

#     error_hidden = np.dot(error_hidden_2, w1_2) * relu_derivative(z1)
#     dw1 = np.dot(error_hidden.T, x) / x.shape[0]
#     db1 = np.sum(error_hidden, axis=0) / x.shape[0]

#     return dw1, db1, dw1_2, db1_2, dw2, db2

def backpropagation(x, y_true, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2):
    # Forward pass
    z1 = np.dot(x, w1.T) + b1
    a1 = relu(z1)
    z1_2 = np.dot(a1, w1_2.T) + b1_2
    a1_2 = relu(z1_2)
    z1_3 = np.dot(a1_2, w1_3.T) + b1_3
    a1_3 = relu(z1_3)
    z2 = np.dot(a1_3, w2.T) + b2
    a2 = softmax(z2)

    # Calculul erorii la stratul de ieșire
    error_output = a2 - y_true

    # Calculul gradienților pentru ponderile și biasurile stratului de ieșire
    dw2 = np.dot(error_output.T, a1_3) / x.shape[0]
    db2 = np.sum(error_output, axis=0) / x.shape[0]

    # Propagarea erorii înapoi la al treilea strat ascuns
    error_hidden_3 = np.dot(error_output, w2) * relu_derivative(z1_3)
    dw1_3 = np.dot(error_hidden_3.T, a1_2) / x.shape[0]
    db1_3 = np.sum(error_hidden_3, axis=0) / x.shape[0]

    # Propagarea erorii înapoi la al doilea strat ascuns
    error_hidden_2 = np.dot(error_hidden_3, w1_3) * relu_derivative(z1_2)
    dw1_2 = np.dot(error_hidden_2.T, a1) / x.shape[0]
    db1_2 = np.sum(error_hidden_2, axis=0) / x.shape[0]

    # Propagarea erorii înapoi la primul strat ascuns
    error_hidden = np.dot(error_hidden_2, w1_2) * relu_derivative(z1)
    dw1 = np.dot(error_hidden.T, x) / x.shape[0]
    db1 = np.sum(error_hidden, axis=0) / x.shape[0]

    return dw1, db1, dw1_2, db1_2, dw1_3, db1_3, dw2, db2
    
# Actualizarea ponderilor și biasurilor pentru 2 straturi ascunse:
# def update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
#     w1 -= learning_rate * dw1
#     b1 -= learning_rate * db1
#     w2 -= learning_rate * dw2
#     b2 -= learning_rate * db2
#     return w1, b1, w2, b2

# def update_weights(w1, b1, w1_2, b1_2, w2, b2, dw1, db1, dw1_2, db1_2, dw2, db2, learning_rate):
#     w1 -= learning_rate * dw1
#     b1 -= learning_rate * db1
#     w1_2 -= learning_rate * dw1_2
#     b1_2 -= learning_rate * db1_2
#     w2 -= learning_rate * dw2
#     b2 -= learning_rate * db2
#     return w1, b1, w1_2, b1_2, w2, b2

def update_weights(w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2, 
                   dw1, db1, dw1_2, db1_2, dw1_3, db1_3, dw2, db2, 
                   learning_rate):
    # Actualizare ponderi și biasuri pentru primul strat ascuns
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1

    # Actualizare ponderi și biasuri pentru al doilea strat ascuns
    w1_2 -= learning_rate * dw1_2
    b1_2 -= learning_rate * db1_2

    # Actualizare ponderi și biasuri pentru al treilea strat ascuns
    w1_3 -= learning_rate * dw1_3
    b1_3 -= learning_rate * db1_3

    # Actualizare ponderi și biasuri pentru stratul de ieșire
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    return w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2

# Bucla de Antrenare pentru 2 straturi ascunse
# def train_network(epochs, learning_rate, train_images, train_labels_onehot, w1, b1, w2, b2):
#     for epoch in range(epochs):
#         # Forward pass
#         y_pred = forward(train_images, w1, b1, w2, b2)

#         # Calculul pierderii
#         loss = cross_entropy_loss(y_pred, train_labels_onehot)

#         # Backpropagation
#         dw1, db1, dw2, db2 = backpropagation(train_images, train_labels_onehot, w1, b1, w2, b2)

#         # Actualizarea ponderilor
#         w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        
#         print(f'Epoch {epoch}, Loss: {loss}')

#     return w1, b1, w2, b2
# def train_network(epochs, learning_rate, train_images, train_labels_onehot, w1, b1, w1_2, b1_2, w2, b2):
#     for epoch in range(epochs):
#         # Forward pass
#         y_pred = forward(train_images, w1, b1, w1_2, b1_2, w2, b2)

#         # Calculul pierderii
#         loss = cross_entropy_loss(y_pred, train_labels_onehot)

#         # Backpropagation
#         dw1, db1, dw1_2, db1_2, dw2, db2 = backpropagation(train_images, train_labels_onehot, w1, b1, w1_2, b1_2, w2, b2)

#         # Actualizarea ponderilor
#         w1, b1, w1_2, b1_2, w2, b2 = update_weights(w1, b1, w1_2, b1_2, w2, b2, dw1, db1, dw1_2, db1_2, dw2, db2, learning_rate)

#         print(f'Epoch {epoch}, Loss: {loss}')

#     return w1, b1, w1_2, b1_2, w2, b2

def train_network(epochs, learning_rate, train_images, train_labels_onehot, 
                  w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2):
    for epoch in range(epochs):
        # Forward pass
        y_pred = forward(train_images, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2)

        # Loss entropy
        loss = cross_entropy_loss(y_pred, train_labels_onehot)

        # Backpropagation
        dw1, db1, dw1_2, db1_2, dw1_3, db1_3, dw2, db2 = backpropagation(
            train_images, train_labels_onehot, 
            w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2
        )

        # Update loss 
        w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2 = update_weights(
            w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2, 
            dw1, db1, dw1_2, db1_2, dw1_3, db1_3, dw2, db2, 
            learning_rate
        )

        print(f'Epoch {epoch}, Loss: {loss}')

    return w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2


# Train model
learning_rate = 0.003
epochs = 3000

#w1, b1, w2, b2 = train_network(epochs, learning_rate, train_images, train_labels_onehot, w1, b1, w2, b2)
#w1, b1, w1_2, b1_2, w2, b2 = train_network(epochs, learning_rate, train_images, train_labels_onehot, w1, b1, w1_2, b1_2, w2, b2)
w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2 = train_network(
    epochs, learning_rate, 
    train_images, train_labels_onehot, 
    w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2
)

#Evaluarea Modelului
# def evaluate_model(X, Y_true, w1, b1, w2, b2):
#     # Forward pass
#     Y_pred = forward(X, w1, b1, w2, b2)
    
#     # Convertirea predicțiilor în etichete clasificate
#     predicted_labels = np.argmax(Y_pred, axis=1)
#     true_labels = np.argmax(Y_true, axis=1)

#     # Calculul acurateței
#     accuracy = np.mean(predicted_labels == true_labels)
#     return accuracy
# def evaluate_model(X, Y_true, w1, b1, w1_2, b1_2, w2, b2):
#     # Forward pass
#     Y_pred = forward(X, w1, b1, w1_2, b1_2, w2, b2)
    
#     # Convertirea predicțiilor în etichete clasificate
#     predicted_labels = np.argmax(Y_pred, axis=1)
#     true_labels = np.argmax(Y_true, axis=1)

#     # Calculul acurateței
#     accuracy = np.mean(predicted_labels == true_labels)
#     return accuracy

def evaluate_model(X, Y_true, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2):
    # Forward pass cu toate cele trei straturi ascunse
    Y_pred = forward(X, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2)
    
    # Converting predictions into classified tags
    predicted_labels = np.argmax(Y_pred, axis=1)
    true_labels = np.argmax(Y_true, axis=1)

    # Accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


# Evaluarea modelului
# test_accuracy = evaluate_model(test_images, test_labels_onehot, w1, b1, w2, b2)
# print(f'Acuratețea pe setul de test: {test_accuracy}')

# test accuracy for 2 hidden layers
#test_accuracy = evaluate_model(test_images, test_labels_onehot, w1, b1, w1_2, b1_2, w2, b2)
# test accuracy for 3 hidden layers
test_accuracy = evaluate_model(test_images, test_labels_onehot, w1, b1, w1_2, b1_2, w1_3, b1_3, w2, b2)

print(f'Accuracy on the test set: {test_accuracy}')
