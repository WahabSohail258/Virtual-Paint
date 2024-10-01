import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_mnist_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]
    
    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

print(f"Training data size: {train_images.shape[0]} images, {train_images.shape[1]}x{train_images.shape[2]} pixels")
print(f"Testing data size: {test_images.shape[0]} images, {test_images.shape[1]}x{test_images.shape[2]} pixels")
print(f"Number of classes: {len(np.unique(train_labels))}")
def train_mnist_model(model, train_images, train_labels, epochs=5):
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)
    return history

model = create_mnist_model()
history = train_mnist_model(model, train_images, train_labels, epochs=5)

print("\nTraining History:")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels, digits=4))

evaluate_model(model, test_images, test_labels)
model.save('mnist_model.h5')
print("Model saved as 'mnist_model.h5'")
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or could not be loaded.")
    
    # Apply binary threshold
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of the digit
    x, y, w, h = cv2.boundingRect(binary_img)
    img = binary_img[y:y+h, x:x+w]

    # Calculate padding to make the image square
    pad_x = max(h - w, 0) // 2
    pad_y = max(w - h, 0) // 2
    
    # Add padding to make the image square
    img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    
    # Resize the image to 28x28 pixels
    img_new = cv2.resize(img, (28, 28))
    
    # Normalize pixel values
    img_new = img_new / 255.0
    
    # Reshape the image to fit the model input
    img_new = img_new.reshape(-1, 28, 28, 1)
    
    print(f"Image shape after preprocessing: {img_new.shape}")
    
    # Display the preprocessed image
    plt.imshow(img_new[0].reshape(28, 28), cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()
    
    return img_new
def predict_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction[0])
    
    plt.imshow(img[0].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.show()
    
    return predicted_label

image_path = './three.png'  

# Predict the image
predicted_label = predict_image(model, image_path)
print(f"Predicted label for the image: {predicted_label}")
def plot_images(images, labels, title):
     plt.figure(figsize=(10, 5))
     for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
     plt.suptitle(title)
     plt.show()

     plot_images(train_images, train_labels, "First 10 Training Images")

     plot_images(test_images, test_labels, "First 10 Testing Images")
