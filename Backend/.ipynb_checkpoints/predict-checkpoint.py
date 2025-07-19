import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2
import sys


def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)        
        if img is None:
            img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
            img = image.img_to_array(img)
            img = img.reshape(28, 28)
        else:
            img = cv2.resize(img, (28, 28))
        img = img / 255.0
        if np.mean(img) > 0.5:  
            img = 1 - img
        img = np.where(img > 0.3, 1.0, 0.0)
        img = img.reshape(1, 28, 28, 1)        
        return img
    except Exception as e:
        print(f"Error dalam preprocessing: {str(e)}")
        return None


def predict_digit(img_path, model):
    try:
        img_array = preprocess_image(img_path)
        if img_array is None:
            print("Gagal memproses gambar")
            return None, 0
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Visualisasi hasil (optional saat debug)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(original_img, cmap='gray')
        plt.title('Gambar Asli')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title(f'Setelah Preprocessing')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        digits = range(10)
        probabilities = prediction[0] * 100
        bars = plt.bar(digits, probabilities)
        bars[predicted_digit].set_color('red')
        plt.title(f'Prediksi: {predicted_digit} ({confidence:.1f}%)')
        plt.xlabel('Digit')
        plt.ylabel('Probabilitas (%)')
        plt.xticks(digits)

        plt.tight_layout()
        plt.show()

        return predicted_digit, confidence
        
    except Exception as e:
        print(f"Error dalam prediksi: {str(e)}")
        return None, 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        model = keras.models.load_model("mnist_cnn_improved.keras")
        predicted_digit, confidence = predict_digit(image_path, model)
        if predicted_digit is not None:
            # Format output ke backend Go
            print(f"{predicted_digit}|{confidence/100:.2f}")
        else:
            print("Error|0.00")
    except Exception as e:
        print(f"Error|0.00\n{str(e)}")
