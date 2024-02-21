import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load FP-GAN generator model from TensorFlow Hub
generator = hub.load("https://tfhub.dev/google/fpgan/celebahq/512/1")

# Function to generate image using FP-GAN and display/save original and generated images
def generate_and_display_images(input_image_path, output_image_path):
    # Load input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0

    # Resize input image to match the expected input shape of the generator
    input_image = tf.image.resize(input_image, (512, 512))

    # Generate image using FP-GAN
    generated_image = generator(input_image[np.newaxis, ...])[0]

    # Convert generated image to uint8 format
    generated_image = (generated_image.numpy() * 255).astype(np.uint8)

    # Display and save original and generated images
    display_and_save_images(input_image, generated_image, output_image_path)

# Function to display original and generated images side by side and save the generated image
def display_and_save_images(original_image, generated_image, output_image_path):
    plt.figure(figsize=(10, 5))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    # Display generated image
    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    plt.imshow(generated_image)
    plt.axis('off')
    
    plt.savefig(output_image_path)  # Save generated image
    plt.show()

# Example usage
input_image_path = "input_image.jpg"
output_image_path = "output_image.jpg"

generate_and_display_images(input_image_path, output_image_path)
