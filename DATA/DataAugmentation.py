from PIL import Image, ImageOps
import os

angles = [12 * i for i in range(30)]
target_size = (256, 256)
extra_top_bottom = 10

def apply_augmentation(folder_path):    
    files_to_augment = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    generated_count = 0

    for filename in files_to_augment:
        filepath = os.path.join(folder_path, filename)
        base_name, ext = os.path.splitext(filename)

        img = Image.open(filepath).convert('RGBA')
        original_width, original_height = img.size

        max_dim = max(original_width, original_height)
        square_img = Image.new('RGBA', (max_dim, max_dim), (255, 255, 255, 255))
        offset_x = (max_dim - original_width) // 2
        offset_y = (max_dim - original_height) // 2
        square_img.paste(img, (offset_x, offset_y), img)

        # Add white stripe at top and bottom
        padded_height = max_dim + 2 * extra_top_bottom
        padded_img = Image.new('RGBA', (max_dim, padded_height), (255, 255, 255, 255))
        padded_img.paste(square_img, (0, extra_top_bottom), square_img)

        # Resize to final size
        resized_img = padded_img.resize(target_size, Image.Resampling.LANCZOS)

        original_resized_rgb = resized_img.convert('RGB')
        original_resized_filename = f"{base_name}_2.jpg" 
        original_resized_rgb.save(os.path.join(folder_path, original_resized_filename), format='JPEG')
        generated_count += 1
        
        for i, angle in enumerate(angles, start=1):
            
            if angle == 0:
                continue 

            canvas = Image.new('RGBA', target_size, (255, 255, 255, 255))
            canvas.paste(resized_img, (0, 0), resized_img)

            rotated = canvas.rotate(angle, expand=False)
            
            white_bg = Image.new('RGBA', rotated.size, (255, 255, 255, 255))
            final_rgba = Image.alpha_composite(white_bg, rotated)

            final_img = final_rgba.convert('RGB')
            new_filename = f"{base_name}_{i+1}.jpg" 
            final_img.save(os.path.join(folder_path, new_filename), format='JPEG')
            generated_count += 1
    
        try:
            os.remove(filepath)
        except OSError as e:
            print(f"Error {filename}: {e}")

train_dir = 'Dataset/train'

apply_augmentation(train_dir)