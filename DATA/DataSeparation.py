import os
import shutil
import random 

# Directories
input_dir = './Molecules'
output_base = 'Dataset' # Output folder where train/, val/, test/ will be created

# Split Ratios
train_split = 0.7
val_split = 0.15
# test_split will be the remainder (0.15)

# --- 1. Prepare Directories ---
for split in ['train', 'val', 'test']:
    # Create output folders if they don't exist
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# --- 2. List Original Files (Each File = One "Group"/Molecule) ---
# Filter only image files
all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))]

# Shuffle the list of files randomly (fixed by the seed)
random.shuffle(all_files)

# --- 3. Split the Files ---
num_total = len(all_files)
num_train = int(train_split * num_total)
num_val = int(val_split * num_total)

# Divide the list based on calculated counts
train_files = all_files[:num_train]
val_files = all_files[num_train:num_train + num_val]
test_files = all_files[num_train + num_val:]

# --- 4. Auxiliary Function for Copying ---
def copy_files_to_split(file_list, split_name):
    dest_dir = os.path.join(output_base, split_name)
    for file in file_list:
        src = os.path.join(input_dir, file)
        dst = os.path.join(dest_dir, file)
        # Copy the file to the destination folder
        shutil.copy2(src, dst)

# --- 5. Copy the Files ---
copy_files_to_split(train_files, 'train')
copy_files_to_split(val_files, 'val')
copy_files_to_split(test_files, 'test')

print(f"Separation completed: Train ({len(train_files)}), Validation ({len(val_files)}), Test ({len(test_files)})")