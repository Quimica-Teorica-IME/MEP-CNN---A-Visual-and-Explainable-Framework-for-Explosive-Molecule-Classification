import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import inspect
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

def compute_saliency_map_classification(model, image_array, classItem, save_path=None):
    image_tensor = tf.convert_to_tensor(image_array[None], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        
        x = image_tensor
        for layer in model.layers[:-1]: 
            if hasattr(layer, 'call') and 'training' in inspect.getfullargspec(layer.call).args:
                x = layer(x, training=False)
            else:
                x = layer(x)

        logits = None
        last_dense_layer = model.layers[-1]
        if isinstance(last_dense_layer, tf.keras.layers.Dense) and \
           last_dense_layer.activation == tf.keras.activations.sigmoid:
            logits = tf.matmul(x, last_dense_layer.kernel) + last_dense_layer.bias
        else:
            logits = x 
        
        if classItem == 1:
            loss = logits[:, 0]
        elif classItem == 0:
            loss = -logits[:, 0]
        else:
            raise ValueError("classItem must be 0 or 1 for binary classification.")

    grads = tape.gradient(loss, image_tensor)

    if grads is None:
        print("Warning: Gradients are None.")
        saliency_np = np.zeros_like(image_array[:,:,0])
    else:
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        saliency_min = tf.reduce_min(saliency)
        saliency_max = tf.reduce_max(saliency)

        if tf.abs(saliency_max - saliency_min) < 1e-8:
            saliency_np = np.zeros_like(saliency.numpy())
        else:
            saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
            saliency_np = saliency.numpy()

    # --- PLOTS (3 images) ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(image_array)
    axs[0].axis("off")
    axs[0].set_title("(a) Input image", fontsize=20)

    axs[1].imshow(image_array)
    axs[1].imshow(saliency_np, cmap='hot', alpha=0.85)
    axs[1].axis("off")
    axs[1].set_title("(b) Overlaid saliency", fontsize=20)

    axs[2].imshow(saliency_np, cmap='hot')
    axs[2].axis("off")
    axs[2].set_title("(c) Saliency map", fontsize=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.jpg', '.svg').replace('.png', '.svg'),
                    bbox_inches='tight', format='svg')
        plt.close()
    else:
        plt.show()

input_dir = "./DATA/Dataset/train"
saliency_output_dir_0 = "DATA/output/class0-rotate"
img_size = (256, 256)
model = load_model('CNN-ESP.keras')

img_size = (256, 256)

os.makedirs(saliency_output_dir_0, exist_ok=True)
# List images
image_files = sorted([
    f for f in os.listdir(input_dir)
    if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

print(f"{len(image_files)} images found in '{input_dir}'.")

# ==== Loop through images ====
for idx, filename in enumerate(image_files):
    if idx < 20:
        input_path = os.path.join(input_dir, filename)

        # Load and prepare image
        img = image.load_img(input_path, target_size=img_size)
        img_array = img_to_array(img).astype('float32') / 255.0

        # Prediction (optional: show)
        pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        # Check if pred has only one value
        if pred.ndim == 0:
            predicted_class = int(pred > 0.5)
        else:
            predicted_class = int((pred > 0.5).astype(int)[0])

        print(f"[{idx+1}] {filename} — Prediction: {predicted_class}")

        # Generate and save saliency map
        if predicted_class == 0:
            output_path = os.path.join(saliency_output_dir_0, f"saliency_{filename}.png")
            compute_saliency_map_classification(model, img_array, predicted_class, save_path=output_path)