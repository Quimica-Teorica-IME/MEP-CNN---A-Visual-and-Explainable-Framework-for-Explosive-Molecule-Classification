import os
import tensorflow as tf
import numpy as np
import itertools
from collections import Counter
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError

base_path = 'DATA/Dataset'
target_size = (256, 256)
BATCH_SIZE = 32
NUM_EPOCHS = 50

HYPERPARAMS_GRID = {
    'dropout_conv': [0.1, 0.3],              
    'dropout_dense': [0.5, 0.7],             
    'l2_reg': [0.0001, 0.001],               
    'learning_rate': [0.00005, 0.0001],      
    'rotation_factor': [0.05, 0.1]           
}

all_combinations = list(itertools.product(*HYPERPARAMS_GRID.values()))
print(f"{len(all_combinations)}")

def load_dataset(split):
    images = []
    labels = []

    split_path = os.path.join(base_path, split)

    for fname in os.listdir(split_path):
        fpath = os.path.join(split_path, fname)
        
        try:
            img = image.load_img(fpath, target_size=target_size)
            
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            
            label = 1 if 'explosive' in fname.lower() else 0
            
            images.append(img_array)
            labels.append(label)

        except UnidentifiedImageError:
            print(f"'{fpath}' corrupted file.")
            continue
        except Exception as e:
            print(f"Error in path '{fpath}': {e}")
            continue

    if not images:
        print(f"No files found in the directory: '{split_path}'.")
        return tf.data.Dataset.from_tensors((tf.constant([]), tf.constant([])))

    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    
    images_tensor = tf.convert_to_tensor(images_np, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels_np, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(32).prefetch(tf.data.AUTOTUNE)
    print(Counter(labels))

    return dataset

train_dataset = load_dataset('train')
val_dataset = load_dataset('val')
test_dataset = load_dataset('test')

strategy = tf.distribute.get_strategy()

def create_compiled_model(params):
    dropout_conv = params['dropout_conv']
    dropout_dense = params['dropout_dense']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']
    rotation_factor = params['rotation_factor']

    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(256, 256, 3)))
        
        model.add(RandomFlip('horizontal_and_vertical'))
        model.add(RandomRotation(rotation_factor))  
        model.add(RandomZoom(0.1))

        model.add(Conv2D(16, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(dropout_conv)) 

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(dropout_conv)) 

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(dropout_conv)) 

        model.add(GlobalAveragePooling2D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_dense)) 
        model.add(Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

best_val_accuracy = 0
best_params = {}
results = []
param_names = list(HYPERPARAMS_GRID.keys())

for i, combination in enumerate(all_combinations):
    current_params = dict(zip(param_names, combination))
    print(f"\n--- Test {i+1}/{len(all_combinations)}: {current_params} ---")
    
    model = create_compiled_model(current_params)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    
    train_loss, train_accuracy = model.evaluate(train_dataset, verbose=0)
    
    print(f"Final Result: Val Accuracy = {val_accuracy:.4f} | Train Accuracy = {train_accuracy:.4f}")
    
    results.append({
        'params': current_params,
        'val_accuracy': val_accuracy,
        'train_accuracy': train_accuracy,
        'generalization_gap': train_accuracy - val_accuracy
    })
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = current_params
        model.save('CNN-ESP_best_grid_model.keras')
        print(">>> Best Model <<<")

print("\n" + "="*50)
print("GRID SEARCH")
print("="*50)

results.sort(key=lambda x: x['val_accuracy'], reverse=True)

print("TOP 3 BEST CONFIGURATIONS (Based on Validation Accuracy):")
for i, res in enumerate(results[:3]):
    print(f"--- RANK {i+1} ---")
    print(f"Params: {res['params']}")
    print(f"Accuracy Validation: {res['val_accuracy']:.4f}")
    print(f"Accuracy Training: {res['train_accuracy']:.4f}")
    print(f"Gap (Overfitting): {res['generalization_gap']:.4f}")

print(best_params)
