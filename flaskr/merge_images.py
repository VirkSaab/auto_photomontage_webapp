import cv2
from numpy import ndarray


def make_collage(images:dict, orientation:str) -> ndarray: # image dims: H, W, C
    for name, image in images.items():
        print(f"[MAIN] name: {name}, shape: {image.shape}")
    if orientation == "portrait":
        final_image = make_portrait_collage(images)
    elif orientation == "landscape":
        final_image = make_landscape_collage(images)
    else:
        raise ValueError("Orientation not understood! choose either portrait or landscape.")
    return final_image

# ============================ PORTRAIT
def portrait_resize_images(images:dict, interpolation=cv2.INTER_CUBIC) -> dict:
    # take minimum width 
    w_min = min(img.shape[1] for _, img in images.items())
    # resizing images 
    return {
        name: cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation = interpolation)
        for name, img in images.items()
    }

def make_portrait_collage(images:dict) -> ndarray:
    images = portrait_resize_images(images)
    for name, image in images.items():
        print(f"name: {name}, shape: {image.shape}")

    final_image = cv2.vconcat(list(images.values()))
    return final_image



# ============================ LANDSCAPE
def landscape_resize_images(images:dict, interpolation=cv2.INTER_CUBIC) -> dict:
    # take minimum height
    h_min = min(img.shape[0] for _, img in images.items())
    # resizing images 
    return {
        name: cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation = interpolation)
        for name, img in images.items()
    }
    
def make_landscape_collage(images:dict) -> ndarray:
    images = landscape_resize_images(images)
    for name, image in images.items():
        print(f"name: {name}, shape: {image.shape}")
    final_image = cv2.hconcat(list(images.values()))
    return final_image

