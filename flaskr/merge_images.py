from numpy import ndarray

def make_collage(images:dict) -> ndarray:
    for image_name, image in images.items():
        # image.shape (Height, Width, Channels)
        print(f"{image_name}: {image.shape}")


