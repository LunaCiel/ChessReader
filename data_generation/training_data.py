import os
import shutil
import re

src_path = "/Users/luna/PycharmProjects/ml-project-2-apa-main/data_generation/data"
dst_path = "/Users/luna/PycharmProjects/ml-project-2-apa-main/training_data"

# remove all files from destination directory
for filename in os.listdir(dst_path):
    file_path = os.path.join(dst_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

files = os.listdir(src_path)

image_and_text_files = [f for f in files if f.endswith(('.jpg', '.png', '.txt'))]

# function to sort files based on the numeric value in their names
def sort_files(file):
    num = re.findall(r'\d+', file)
    return int(num[0]) if num else 0

# sort the files
image_and_text_files.sort(key=sort_files)

# ensure destination directory exists
os.makedirs(dst_path, exist_ok=True)

for file in image_and_text_files[:3000]:
    shutil.copy(os.path.join(src_path, file), dst_path)