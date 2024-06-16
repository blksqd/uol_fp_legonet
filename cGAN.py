import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Configurations
batch_size = 64
num_channels = 3  # RGB images have 3 channels
num_classes = 2
image_size = (64, 64)  # Ensure this matches your patch size
latent_dim = 128

# Paths to data
patch_dir = '../LEGO/data/patches/'  # Directory containing patch images
csv_path = '../LEGO/data/csv/patch_metadata.csv'  # Path to CSV file with metadata

# Start time tracking
start_time = time.time()

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(csv_path)
print(f"Metadata loaded. {metadata.shape[0]} entries found.")
print(f"Time taken to load metadata: {time.time() - start_time:.2f} seconds")

# Encode labels to integers then to one-hot
print("Encoding labels...")
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(metadata['class'])
one_hot_labels = to_categorical(integer_encoded, num_classes=num_classes)
print(f"Labels encoded. Number of classes: {num_classes}")
print(f"Time taken to encode labels: {time.time() - start_time:.2f} seconds")

# Function to get the full path of the patch
def get_patch_path(row):
    patch_filename = f"{row['patch_id']}.png"
    return os.path.join(patch_dir, patch_filename)

# Load and process patch images
def process_patch(patch_path):
    img = load_img(patch_path, target_size=image_size, color_mode='rgb')
    img_array = img_to_array(img).astype("float32") / 255.0
    return img_array

# Load all patches and corresponding labels
print("Loading and processing patches...")
patch_paths = [get_patch_path(row) for _, row in metadata.iterrows()]
patches = np.array([process_patch(patch_path) for patch_path in patch_paths if os.path.exists(patch_path)])
labels = one_hot_labels[:len(patches)]  # Align labels with loaded patches
print(f"Patches loaded and processed. {len(patches)} patches found.")
print(f"Time taken to load patches: {time.time() - start_time:.2f} seconds")

# Ensure the images have 4 dimensions (batch_size, height, width, channels)
patches = np.reshape(patches, (-1, image_size[0], image_size[1], num_channels))

# Create tf.data.Dataset
print("Creating TensorFlow dataset...")
dataset = tf.data.Dataset.from_tensor_slices((patches, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
print(f"Dataset created with batch size {batch_size}.")
print(f"Time taken to create dataset: {time.time() - start_time:.2f} seconds")

# Print data shapes for confirmation
print(f"Shape of patches: {patches.shape}")
print(f"Shape of labels: {labels.shape}")

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes

print(f"Generator input channels: {generator_in_channels}")
print(f"Discriminator input channels: {discriminator_in_channels}")

# Create the discriminator with added dropout and input noise
print("Building discriminator...")
discriminator = tf.keras.Sequential(
    [
        layers.InputLayer((image_size[0], image_size[1], discriminator_in_channels)),
        layers.GaussianNoise(0.1),  # Adding noise to the inputs
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Dropout(0.3),  # Adding dropout to prevent overfitting
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Dropout(0.3),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)
print("Discriminator built.")
print(f"Time taken to build discriminator: {time.time() - start_time:.2f} seconds")

# Create the generator with an additional layer to enhance capacity
print("Building generator...")
generator = tf.keras.Sequential(
    [
        layers.InputLayer((generator_in_channels,)),
        layers.Dense((image_size[0] // 4) * (image_size[1] // 4) * 128),
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Reshape((image_size[0] // 4, image_size[1] // 4, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"),  # Additional layer
        layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
        layers.Conv2D(num_channels, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
print("Generator built.")
print(f"Time taken to build generator: {time.time() - start_time:.2f} seconds")

class ConditionalGAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        # Ensure one_hot_labels is float32
        one_hot_labels = tf.cast(one_hot_labels, tf.float32)

        # Expand labels to match the shape of images for concatenation
        image_one_hot_labels = tf.reshape(one_hot_labels, (-1, 1, 1, num_classes))
        image_one_hot_labels = tf.tile(image_one_hot_labels, [1, image_size[0], image_size[1], 1])

        # Sample random points in the latent space and concatenate the labels
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images
        generated_images = self.generator(random_vector_labels)

        # Ensure generated images have the same size as real images
        generated_images = tf.image.resize(generated_images, image_size)

        # Combine them with real images
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], axis=-1)
        real_image_and_labels = tf.concat([tf.cast(real_images, tf.float32), image_one_hot_labels], axis=-1)
        combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_images = tf.image.resize(fake_images, image_size)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], axis=-1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        # Print the current loss values for generator and discriminator
        tf.print("Generator loss:", self.gen_loss_tracker.result())
        tf.print("Discriminator loss:", self.disc_loss_tracker.result())

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        latent_dim = config['latent_dim']
        discriminator = tf.keras.Sequential(
            [
                layers.InputLayer((image_size[0], image_size[1], discriminator_in_channels)),
                layers.GaussianNoise(0.1),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Dropout(0.3),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Dropout(0.3),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )

        generator = tf.keras.Sequential(
            [
                layers.InputLayer((latent_dim + num_classes,)),
                layers.Dense((image_size[0] // 4) * (image_size[1] // 4) * 128),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Reshape((image_size[0] // 4, image_size[1] // 4, 128)),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),  # Changed negative_slope to alpha
                layers.Conv2D(num_channels, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        return cls(discriminator, generator, latent_dim)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        config = {
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)

        self.generator.save_weights(filepath + '_generator.weights.h5', overwrite=overwrite)
        self.discriminator.save_weights(filepath + '_discriminator.weights.h5', overwrite=overwrite)

    @classmethod
    def load(cls, filepath):
        with open(filepath + '_config.json', 'r') as f:
            config = json.load(f)

        model = cls.from_config(config['config'])
        model.generator.load_weights(filepath + '_generator.weights.h5')
        model.discriminator.load_weights(filepath + '_discriminator.weights.h5')
        return model

# Instantiate and compile the ConditionalGAN model
print("Initializing Conditional GAN...")
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)
print("Conditional GAN initialized.")
print(f"Time taken to initialize GAN: {time.time() - start_time:.2f} seconds")

# Lists to store loss values for plotting
g_losses = []
d_losses = []

# Custom training loop to capture losses
for epoch in range(3):  # Replace 3 with the number of epochs you want
    print(f"Epoch {epoch + 1}/{3}")  # Update to reflect the correct number of epochs
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        loss = cond_gan.train_step((x_batch_train, y_batch_train))
        g_losses.append(loss['g_loss'].numpy())
        d_losses.append(loss['d_loss'].numpy())
    print(f"Epoch {epoch + 1} completed. Generator loss: {loss['g_loss']}, Discriminator loss: {loss['d_loss']}")

print("Training completed.")
print(f"Total time taken for training: {time.time() - start_time:.2f} seconds")

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Losses for Generator and Discriminator')
plt.legend()
plt.show()

# Save the model
print("Saving the model...")
cond_gan.save('cond_gan_model')
print("Model saved.")
print(f"Time taken to save the model: {time.time() - start_time:.2f} seconds")

# Load the model for verification
print("Loading the model for verification...")
loaded_model = ConditionalGAN.load('cond_gan_model')
print("Model loaded.")
print(f"Time taken to load the model: {time.time() - start_time:.2f} seconds")

# Recompile the model after loading
print("Recompiling the loaded model...")
loaded_model.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)
print("Model recompiled.")
print(f"Time taken to recompile the model: {time.time() - start_time:.2f} seconds")

# Build the discriminator by specifying input shape
loaded_model.discriminator.build(input_shape=(None, image_size[0], image_size[1], discriminator_in_channels))

# Verify the model structure
print("Discriminator summary:")
loaded_model.discriminator.summary()

# Build the generator by specifying input shape
loaded_model.generator.build(input_shape=(None, latent_dim + num_classes))

# Verify the generator structure
print("Generator summary:")
loaded_model.generator.summary()
