import cv2
import numpy as np
from tqdm import tqdm
from rle import decode_rle, convert_binary_mask_to_rle

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def create_coco_categories(classes):
    """ Create categories section for COCO JSON. """
    categories = [{"id": idx, "name": cls} for idx, cls in enumerate(classes)]
    return categories

def decode_segmentation_and_generate_annotations(data_frame, image_id, filepath, classes):
    """ Decode segmentation data and prepare annotations for a single image. """
    annotations = []
    file_id = ('_'.join(
        (filepath.split("/")[-3] + "_" + filepath.split("/")[-1]).split("_")[:-4]))
    height_slice = int(filepath.split("/")[-1].split("_")[3])
    width_slice = int(filepath.split("/")[-1].split("_")[2])

    ids = data_frame.index[data_frame['id'] == file_id].tolist()
    if len(ids) > 0:
        for idx in ids:
            segmentation_mask = decode_rle(
                data_frame.iloc[idx]['segmentation'], (height_slice, width_slice))
            for contour in cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]:
                mask_image = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                cv2.drawContours(mask_image, [contour], -1, 255, -1)
                encoded_segmentation = convert_binary_mask_to_rle(mask_image)
                ys, xs = np.where(mask_image)
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                annotations.append({
                    'segmentation': encoded_segmentation,
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # (x, y, w, h) format
                    'area': mask_image.sum(),
                    'image_id': image_id,
                    'category_id': classes.index(data_frame.iloc[idx]['class']),
                    'iscrowd': 0,
                    'id': len(annotations)
                })
    return annotations

def create_coco_images(filepaths):
    """ Create images section for COCO JSON. """
    images = []
    for i, filepath in enumerate(tqdm(filepaths, desc="Processing images")):
        file_name = '/'.join(filepath.split("/")[3:])
        height = int(filepath.split("/")[-1].split("_")[3])
        width = int(filepath.split("/")[-1].split("_")[2])
        images.append({
            "id": i + 1,
            "file_name": file_name,
            "width": width,
            "height": height
        })
    return images

def create_coco_format_json(data_frame, classes, filepaths):
    """ Main function to create COCO formatted JSON. """
    categories = create_coco_categories(classes)
    images = create_coco_images(filepaths)
    annotations = []
    count = 0
    
    for image in tqdm(images, desc="Generating annotations"):
        image_id = image['id']
        filepath = image['file_name']
        
        file_id = ('_'.join(
            (filepath.split("/")[-3] + "_" + filepath.split("/")[-1]).split("_")[:-4]))
        height_slice = int(filepath.split("/")[-1].split("_")[3])
        width_slice = int(filepath.split("/")[-1].split("_")[2])
    
        ids = data_frame.index[data_frame['id'] == file_id].tolist()
        if len(ids) > 0:
            for idx in ids:
                segmentation_mask = decode_rle(
                    data_frame.iloc[idx]['segmentation'], (height_slice, width_slice))
                for contour in cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]:
                    mask_image = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                    cv2.drawContours(mask_image, [contour], -1, 255, -1)
                    encoded_segmentation = convert_binary_mask_to_rle(mask_image)
                    ys, xs = np.where(mask_image)
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    annotations.append({
                        'segmentation': encoded_segmentation,
                        'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # (x, y, w, h) format
                        'area': mask_image.sum(),
                        'image_id': image_id,
                        'category_id': classes.index(data_frame.iloc[idx]['class']),
                        'iscrowd': 0,
                        'id': count
                    })
                    count += 1
        
    return {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }