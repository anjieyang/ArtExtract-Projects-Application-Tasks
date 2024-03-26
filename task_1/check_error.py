from pathlib import Path
from PIL import Image

def check_broken_images(dataset_path):
    """
    Checks which image files in the dataset might cause the 'OSError: broken data stream when reading image file' error.
    Returns a list of paths to problematic images.
    """
    broken_images = []

    for image_path in Path(dataset_path).rglob('*.*'):
        try:
            with open(str(image_path), 'rb') as f:
                Image.open(f).convert('RGB').load()
        except OSError as e:
            if 'broken data stream' in str(e):
                broken_images.append(str(image_path))

    return broken_images

dataset_path = './wikiart/Post_Impressionism'
broken_images = check_broken_images(dataset_path)

if broken_images:
    print("The following images have a 'broken data stream' issue:")
    for img_path in broken_images:
        print(img_path)
else:
    print("No images with a 'broken data stream' issue were found.")