import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Dense
import cv2
import matplotlib.pyplot as plt

# تعریف ابعاد تصویر
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# تعریف ژنراتور
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, inputs):
        return self.model(inputs)

# تعریف تابع زیان
def loss_fn(real_images, fake_images):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_images), real_images)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_images), fake_images)
    return real_loss + fake_loss

# تعریف بهینه سازها
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# خواندن تصویر
image = cv2.imread('input_image.jpg')
# تغییر اندازه تصویر به ابعاد مشخص شده
image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
# تبدیل تصویر به ماتریس نرمالایز شده
image_normalized = image_resized / 255.0
# اضافه کردن بعد اول به تصویر
image_normalized = tf.expand_dims(image_normalized, axis=0)

# ایجاد مولد
generator = Generator()

# تولید تصویر جعلی
fake_image = generator(image_normalized)

# نمایش تصویر ورودی و تصویر جعلی
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(tf.squeeze(fake_image).numpy(), cmap='gray')
plt.title('Generated Image')
plt.show()
