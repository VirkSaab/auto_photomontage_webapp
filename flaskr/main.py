"""
Auto Photomontage image processing file
Author: Jitender Singh Virk
Github ID: VirkSaab
Last updated: 2 Jan 2021
"""

from typing import Tuple
import cv2, os
import numpy as np
from collections import OrderedDict


def make_collage(images:dict) -> np.ndarray: # image dims: H, W, C
    # for name, image in images.items():
        # print(f"[MAIN] name: {name}, shape: {image.shape}")
    mont = MakeMontage(images)
    return mont.output #final_image

def minmaxscaler(x): return (x - x.min()) / (x.max() - x.min())

def smooth_blend(im1, im2, horizontal=True):
    if horizontal:
        alpha = np.linspace(1, 0, num=im1.shape[1])
        alpha = np.repeat(alpha.reshape(1, -1), im1.shape[0], axis=0) # make height
    else:
        alpha = np.linspace(1, 0, num=im1.shape[0])
        alpha = np.repeat(alpha.reshape(1, -1), im1.shape[1], axis=0).T # make width
    alpha = np.repeat(alpha.reshape(*alpha.shape, 1), im1.shape[-1], axis=-1) # make channels
    out = minmaxscaler((alpha * (im1/255)) + ((1 - alpha)* im2/255))
    out = (out * 255).astype(np.uint8)
    return out


class MakeMontage:
    def __init__(self, images:dict, interpolation=cv2.INTER_CUBIC, roi_margin:int=10, merge_pct=8):
        self.images = images
        self.interpolation = interpolation
        self.num_images = len(self.images)
        self.roi_margin = roi_margin # Margin to add around ROI image
        self.merge_pct = merge_pct / 100. # Percentage of image area to merge with another image

        self.E_imp_dict = {}
        self.E_rep_dict = {}
        self.E_obj_dict = {}
        self.E_trans_dict = {}
        self.weights = {"rep":1., "imp":10., "trans":5., "obj":1.}

    @property
    def output(self) -> np.ndarray:
        # * E_imp
        self.images, self.E_imp_dict = self.compute_image_roi(self.images)
        # * E_rep
        self.E_rep_dict = self.compute_image_representation(self.images)
        # * E_obj
        self.E_obj_dict = self.compute_image_objectness(self.images)
        # * E_trans
        self.E_trans_dict = self.compute_image_transition(self.images)

        # Set Energies
        self.energies = {name: self.get_image_energy(name) for name in self.images.keys()}
        # print("self.energies:", len(self.energies), self.energies)

        # Sort images according to energy values in descending order. (min energy first)
        self.energies = OrderedDict(sorted(self.energies.items(), key=lambda item: item[1]))
        # print("\n[SORTED] self.energies:", len(self.energies), self.energies)
        self.images = OrderedDict({k:self.images[k] for k in self.energies.keys()})
        # print("[SORTED image sizes]", len(self.images), {k:v.shape for k,v in self.images.items()})

        if len(self.images) % 2 != 0: # If odd number of images are present
            odd_pair = {}
            for i, (name, img) in enumerate(self.images.items()):
                odd_pair[name] = img
                if i == 1: break

            # Delete odd pair images from images list
            for name in odd_pair.keys():
                del self.images[name]

            odd_pair_name = "__".join(list(odd_pair.keys()))
            odd_img = self.merge_one_pair({odd_pair_name: list(odd_pair.values())}, resolution="max")
            self.images.update(odd_img)

        # Get pairs of images
        image_pairs, self.unpaired_images = self.get_image_pairs(self.images)
        # print("image_pairs:", len(image_pairs))
        # print("self.unpaired_images", len(self.unpaired_images), self.unpaired_images.keys())

        # Combine pairs of images
        self.final_image, unpaired = self.merge_pairs(image_pairs)
        if len(unpaired) != 0:
            self.final_image.update(unpaired)
            image_pairs, _ = self.get_image_pairs(self.final_image)
            self.final_image = self.merge_one_pair(image_pairs[0], resolution="max")
        # print("unpaired", len(unpaired), unpaired)

        # return image only, not dict
        self.final_image = list(self.final_image.values())[0]
        return self.final_image

    def get_image_energy(self, name:str) -> float:
        imp = self.E_imp_dict[name] * self.weights["imp"]
        rep = self.E_rep_dict[name] * self.weights["rep"]
        obj = self.E_obj_dict[name] * self.weights["obj"]
        trans = self.E_trans_dict[name] * self.weights["trans"]
        return sum([imp, rep, obj, trans])

    def get_image_pairs(self, images:dict) -> Tuple[list, dict]:
        unpaired = {}
        if len(images) % 2 != 0:
            _pairs = len(images) - 1
            unpaired_name = list(images.keys())[-1]
            unpaired[unpaired_name] = images[unpaired_name]
            del images[unpaired_name]
        else: _pairs = len(images)
        sorted_names = list(images.keys())
        pairs = [
            {'__'.join((sorted_names[i], sorted_names[i+1])):
            [images[sorted_names[i]], images[sorted_names[i+1]]]}
            for i in range(0, _pairs, 2)
        ]
        return pairs, unpaired

    def merge_one_pair(self, pair:dict, resolution:str="min") -> dict:
        pair_name = list(pair.keys())[0]
        pair = pair[pair_name]

        img1_height, img1_width, _ = pair[0].shape
        img2_height, img2_width, _ = pair[1].shape
        max_height = max(img1_height, img2_height)
        max_width  = max(img1_width, img2_width)

        h_ratio = abs(max_height / (img1_width + img2_width) - 0.75) # 0.75 is 4:3 resolution
        w_ratio = abs((img1_height + img2_height)  / max_width - 0.75)

        if h_ratio < w_ratio: # Horizontal Combination
            pair = self.horizontal_combination(pair, resolution)

        else: # Vertical Combination
            pair = self.vertical_combination(pair, resolution)
        return {pair_name: pair}

    def horizontal_combination(self, pair:list, resolution:str="min"):
        left_oimg, right_oimg = pair[0], pair[1]

        # Compute new height and width for merging
        left_height, left_width, _ = left_oimg.shape
        right_height, right_width, _ = right_oimg.shape

        height = eval(f"{resolution}(left_height, right_height)")
        left_width = int(left_width * height / left_height)
        right_width = int(right_width * height / right_height)

        # Resize
        # print("SIZES:", left_oimg.shape, right_oimg.shape)
        left_oimg = cv2.resize(left_oimg, (left_width, height), interpolation=self.interpolation)
        right_oimg = cv2.resize(right_oimg, (right_width, height), interpolation=self.interpolation)
        # print("\n[RESIZED] SIZES:", left_oimg.shape, right_oimg.shape)

        merge_mark = int(min(left_oimg.shape[1], right_oimg.shape[1]) * self.merge_pct) # left width
        left_img = left_oimg[:, :-merge_mark, :]
        right_img = right_oimg[:, merge_mark:, :]

        # patches to merge
        left_patch = left_oimg[:, -merge_mark:, :]
        right_patch = right_oimg[:, :merge_mark, :]
        # print("PATCHES:", left_patch.shape, right_patch.shape)
        merged = smooth_blend(left_patch, right_patch)
        return cv2.hconcat([left_img, merged, right_img])

    def vertical_combination(self, pair:list, resolution:str="min"):
        top_oimg, bottom_oimg = pair[0], pair[1]

        # Compute new height and width for merging
        top_height, top_width, _ = top_oimg.shape
        bottom_height, bottom_width, _ = bottom_oimg.shape

        width = eval(f"{resolution}(top_width, bottom_width)")
        top_height = int(top_height * width / top_width)
        bottom_height = int(bottom_height * width / bottom_width)

        # Resize
        top_oimg = cv2.resize(top_oimg, (width, top_height), interpolation=self.interpolation)
        bottom_oimg = cv2.resize(bottom_oimg, (width, bottom_height), interpolation=self.interpolation)

        merge_mark = int(min(top_oimg.shape[0], bottom_oimg.shape[0]) * self.merge_pct) # top height
        top_img = top_oimg[:-merge_mark, ...]
        bottom_img = bottom_oimg[merge_mark:,...]

        # patches to merge
        top_patch = top_oimg[-merge_mark:,...]
        bottom_patch = bottom_oimg[:merge_mark,...]
        # print("PATCHES:", left_patch.shape, right_patch.shape)
        merged = smooth_blend(top_patch, bottom_patch, horizontal=False)
        return cv2.vconcat([top_img, merged, bottom_img])

    def merge_pairs(self, pairs:list) -> Tuple[dict, dict]:
        all_unpaired = {}
        while True:
            if len(pairs) == 1: return self.merge_one_pair(pairs[0]), all_unpaired
            _pairs = {}
            for pair in pairs: _pairs.update(self.merge_one_pair(pair))
            pairs, unpaired = self.get_image_pairs(_pairs)
            all_unpaired.update(unpaired)

    def compute_image_representation(self, images) -> dict: # E_rep
        _output = {}
        for img_name in images.keys():
            img = images[img_name]
            _output[img_name] = img.std()
            # Histogram
            # hists = np.mean([
            #     np.histogram(img[:,:,c], bins=256, range=(0, 256))[0].var() for c in range(img.shape[-1])
            # ])
            # print("HIST VAL:", hists)
        return _output

    def compute_image_roi(self, images:dict, min_size=(128,128)) -> Tuple[dict, dict]:
        # Save original images sizes for E_rep computation
        original_image_sizes = {name:img.shape for name, img in images.items()}
        E_imp_dict = {}
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        for img_name in images.keys():
            img = images[img_name].copy()
            (success, saliencyMap) = saliency.computeSaliency(img)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            kernel = np.ones((7,7), np.uint8)
            dilated = cv2.dilate(threshMap, kernel, iterations=3)
            x,y,w,h = cv2.boundingRect(dilated)
            x -= self.roi_margin
            y -= self.roi_margin
            w += self.roi_margin
            h += self.roi_margin
            img = img[y:y+h, x:x+w, :]

            # Crop image to salient region
            if success and (img.shape[0] >= min_size[0]) and (img.shape[1] >= min_size[1]):
                images[img_name] = img

            # ----------------- Saliency debuging ----------------------------------
            # dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(dilated, (x,y), (x+w, y+h), (0,255,0), 5)
            # # Crop image to salient region
            # if success and (img.shape[:-1] >= min_size):
            #     images[img_name] = dilated
            # -------------------------------------------------------------

            # Compute E_imp
            oh, ow, _ = original_image_sizes[img_name]
            h, w, _ = images[img_name].shape
            E_imp_dict[img_name] = ((h * w) / (oh * ow)) * 100

        return images, E_imp_dict

    def compute_image_objectness(self, images:dict) -> dict:
        # Face detection
        faces_ratio_dict = {}
        # 'flaskr/static/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

        for img_name in images.keys():
            img = images[img_name]
            oh, ow, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            faces_ratio = sum([(h * w) / (oh * ow) for (x, y, w, h) in faces])
            # # Draw rectangle around the faces
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # if any face near too corner or sides, increase energy to move it to sides of final montage
            for (x, y, w, h) in faces:
                if x - self.roi_margin < img.shape[1] * self.merge_pct:
                    faces_ratio *= 100
                if y - self.roi_margin < img.shape[0] * self.merge_pct:
                    faces_ratio *= 100

            faces_ratio_dict[img_name] = faces_ratio

        return faces_ratio_dict

    def compute_image_transition(self, images):
        e_trans_dict = {}
        for img_name in images.keys():
            img = images[img_name]
            toprow = np.mean(sum(img[0,...] / 255.))
            leftcol = np.mean(sum(img[:, 0, :] / 255.))
            e_trans_dict[img_name] = toprow + leftcol
        return e_trans_dict

    # ! DID NOT WORKED WELL. POOR DETECTION:
    # def compute_image_objectness(self, images:dict, max_detections:int=100) -> dict:
    #     # https://www.programmersought.com/article/98723609770/
    #     saliency = cv2.saliency.ObjectnessBING_create()
    #     saliency.setTrainingPath("flaskr/static/ObjectnessTrainedModel")
    #     for img_name in images.keys():
    #         img = images[img_name]
    #         (success, saliencyMap) = saliency.computeSaliency(img)
    #         numDetections = saliencyMap.shape[0]
    #         for i in range(0, min(numDetections, max_detections)):
    #             (startX, startY, endX, endY) = saliencyMap[i].flatten()
    #             output = img.copy()
    #             color = np.random.randint(0, 255, size=(3,))
    #             color = [int(c) for c in color]
    #             cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
    #         if success:
    #             images[img_name] = output
    #     return images
