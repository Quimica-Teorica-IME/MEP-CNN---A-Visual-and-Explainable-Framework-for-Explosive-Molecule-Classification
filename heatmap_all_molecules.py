import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import inspect
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# ==== Function to generate saliency map (modified) ====
model = load_model('CNN-ESP.keras')

explosive_fig_counter = 1
non_explosive_fig_counter = 1

def compute_saliency_map_classification(model, image_array, classItem, structure_image, molecule_name, save_path=None):
    global explosive_fig_counter, non_explosive_fig_counter
    
    image_tensor = tf.convert_to_tensor(image_array[None], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        
        x = image_tensor
        for layer in model.layers[:-1]: 
            if hasattr(layer, 'call') and 'training' in inspect.getfullargspec(layer.call).args:
                x = layer(x, training=False)
                #print('training=False')
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
        print("Warning: Gradients are None. This may indicate an issue with the model or tape tracking.")
        saliency_np = np.zeros_like(image_array[:,:,0])
    else:
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        saliency_min = tf.reduce_min(saliency)
        saliency_max = tf.reduce_max(saliency)
        
        if tf.abs(saliency_max - saliency_min) < 1e-8: 
            print("Warning: The saliency map has very little variation, likely all zeros or constant.")
            saliency_np = np.zeros_like(saliency.numpy()) 
        else:
            saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
            saliency_np = saliency.numpy()

    h, w, _ = structure_image.shape
    aspect_ratio = w / h
    
    fig, axs = plt.subplots(1, 4, figsize=(18 + aspect_ratio*6, 6))

    if classItem == 1:
        fig_num = explosive_fig_counter + 53
        fig_caption = f"Figure S{fig_num} - {molecule_name}."
        explosive_fig_counter += 1
    else:
        fig_num = non_explosive_fig_counter + 1
        fig_caption = f"Figure S{fig_num} - {molecule_name}."
        non_explosive_fig_counter += 1

    axs[0].imshow(structure_image)
    axs[0].axis("off")
    axs[0].set_title("(a)", fontsize=24)

    axs[1].imshow(image_array)
    axs[1].axis("off")
    axs[1].set_title("(b)", fontsize=24)

    axs[2].imshow(image_array)
    axs[2].imshow(saliency_np, cmap='hot', alpha=0.85)
    axs[2].axis("off")
    axs[2].set_title(f"(c)", fontsize=24)

    axs[3].imshow(saliency_np, cmap='hot')
    axs[3].axis("off")
    axs[3].set_title("(d)", fontsize=24)

    #plt.subplots_adjust(bottom=0.12)
    plt.figtext(0.5, 0.01, f"{fig_caption} - (a) Molecule structure; (b) Input image; (c) Overlaid saliency map; (d) Pure saliency map.", 
                ha='center', fontsize=35, wrap=True)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    #plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.jpg', '.svg').replace('.png', '.svg'), bbox_inches='tight', format='svg')
        plt.close()
    else:
        plt.show()


# ==== Main loop ====
input_dir = "./DATA/Structures"
saliency_output_dir_0 = "output/class-0"
saliency_output_dir_1 = "output/class-1"
img_size = (256, 256)

os.makedirs(saliency_output_dir_0, exist_ok=True)
os.makedirs(saliency_output_dir_1, exist_ok=True)

all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
structure_files = [f for f in all_files if '-SP-MP2-DEF2-TZVP' not in f]
saliency_files = [f for f in all_files if '-SP-MP2-DEF2-TZVP' in f]

file_pairs = {}
for filename in saliency_files:
    base_name = os.path.splitext(filename)[0].replace('-SP-MP2-DEF2-TZVP', '')
    base_name_cleaned = base_name.replace('-explosive', '').replace('explosive', '').replace('.png', '')
    
    structure_filename = None
    for s_file in structure_files:
        
        structure_file = s_file.replace('-explosive','').replace('.png','')
        s_base_name = os.path.splitext(structure_file)[0]
        
        if s_base_name.lower() == base_name_cleaned.lower() or s_base_name.lower() + '-explosive' == base_name_cleaned.lower():
            structure_filename = s_file
            break
            
    if structure_filename:
        file_pairs[filename] = structure_filename
        
print(f"File pairs: {len(file_pairs)}.")

# ==== Loop through images ====
for idx, filename in enumerate(file_pairs.keys()):
    #try:
        #if idx < 10:
            input_path = os.path.join(input_dir, filename)
            structure_path = os.path.join(input_dir, file_pairs[filename])

            img = image.load_img(input_path, target_size=img_size)
            img_array = img_to_array(img).astype('float32') / 255.0

            structure_img = image.load_img(structure_path)
            structure_img_array = img_to_array(structure_img) / 255.0

            molecule_name = os.path.splitext(os.path.basename(structure_path))[0]
            
            if molecule_name.lower().endswith('-explosive'):
                molecule_name = molecule_name[:-10]
            pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]

            if pred.ndim == 0:
                predicted_class = int(pred > 0.5)
            else:
                predicted_class = int((pred > 0.5).astype(int)[0])

            print(f"[{idx+1}] {filename} — Prevision: {predicted_class}")

            if predicted_class == 0:
                output_path = os.path.join(saliency_output_dir_0, f"Fig {idx} - saliency_{filename}")
                compute_saliency_map_classification(model, img_array, predicted_class, structure_img_array, molecule_name, save_path=output_path)
            
            if predicted_class == 1:
                output_path = os.path.join(saliency_output_dir_1, f"Fig {idx} - saliency_{filename}")
                compute_saliency_map_classification(model, img_array, predicted_class, structure_img_array, molecule_name, save_path=output_path)
    #except:
     #   print(f'error to convert figure {filename}')
