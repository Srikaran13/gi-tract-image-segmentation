import os
import pandas as pd
import numpy as np
import itertools
import cv2
from glob import glob
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore')
# from ..config import BASE_PATH, CLASSES


BASE_PATH = '//Users/srikaranreddy/Desktop/Spring Semester/Computer Vision 6.8300/cv-project/gi-tract-image-segmentation'
CLASSES = ['small_bowel', 'large_bowel', 'stomach']
IMAGE_SIZE = (128, 128)

def get_scan_file_path(base_dir, scan_id):
    """
    Helper function to derive the 
    full file path for a given scan ID
    """
    id_parts = scan_id.split("_")
    case_part = id_parts[0]
    day_part = "_".join(id_parts[:2])
    scan_prefix = "_".join(id_parts[2:])
    scan_directory = os.path.join(base_dir, case_part, day_part, "scans")
    matching_files = glob(f"{scan_directory}/{scan_prefix}*")  # Expecting a single match
    return matching_files[0]


class SegmentationDataset():
    def __init__(self, dataset_dir, csv_file_path):
        self.dataset_dir = dataset_dir
        self.train_csv = pd.read_csv(csv_file_path)
        self.processed_df = self.preprocess(self.train_csv)
        # self.categories = self.create_coco_categories(CLASSES)
        # self.images = self.create_coco_images(self.processed_df)
        # self.annotations = self.create_annotations(self.processed_df, self.images)


    def preprocess(self, df):

        df['case'] = df['id'].apply(lambda id_str: id_str.split('_')[0][4:])
        df['day'] = df['id'].apply(lambda id_str: id_str.split('_')[1][3:])
        df['slice'] = df['id'].apply(lambda id_str: id_str.split('_')[-1])
        df['file_path'] = df['id'].apply(lambda id_str: get_scan_file_path(self.dataset_dir, id_str))

        df['file_name'] = df['file_path'].apply(lambda path: os.path.basename(path))
        df['composite_id'] = df.apply(lambda row: f"{row['case']}_{row['day']}_{row['file_name']}", axis=1)

        df['image_height'] = df['file_name'].apply(lambda name: int(name.split('_')[2]))
        df['image_width'] = df['file_name'].apply(lambda name: int(name.split('_')[3]))
        df['resolution'] = df.apply(lambda row: f"{row['image_height']}x{row['image_width']}", axis=1)

        masked_df = df[df['segmentation'].notnull()]
        masked_df["segmentation"] = masked_df["segmentation"].astype("str")
        masked_df = masked_df.reset_index(drop=True)

        return masked_df

    def create_coco_categories(self, classes):
        """ Create categories section for COCO JSON. """
        categories = [{"id": idx, "name": cls} for idx, cls in enumerate(classes)]
        return categories

    def create_coco_images(self, df):
        images = []
        filepaths = df.file_path.unique().tolist()

        for i, filepath in enumerate(tqdm(filepaths, desc="Processing images")):
            file_name = '/'.join(filepath.split("/")[2:])
            height = int(filepath.split("/")[-1].split("_")[3])
            width = int(filepath.split("/")[-1].split("_")[2])
            images.append({
                "id": i + 1,
                "file_name": file_name,
                "width": width,
                "height": height
            })
        return images

    def decode_rle(self, mask_rle, shape):
        '''
        Decode an RLE-encoded mask into a 2D numpy array.

        Parameters:
            mask_rle (str): Run-length encoding of the mask ('start length' pairs)
            shape (tuple): The (height, width) dimensions of the output array

        Returns:
            numpy.ndarray: A 2D array where 1s represent the mask and 0s represent the background
        '''
        # Split the RLE string into a list of strings
        encoded_pixels = mask_rle.split()
        
        # Extract start positions and lengths for the mask
        starts = np.asarray(encoded_pixels[0::2], dtype=int) - 1  # Convert to 0-based indexing
        lengths = np.asarray(encoded_pixels[1::2], dtype=int)
        
        # Calculate the end positions for each segment of the mask
        ends = starts + lengths
        
        # Initialize a flat array for the image
        flat_image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        
        # Apply the mask to the flat image array
        for start, end in zip(starts, ends):
            flat_image[start:end] = 1
        
        # Reshape the flat image array to 2D
        return flat_image.reshape(shape)

    def convert_binary_mask_to_rle(self, binary_mask):
        '''
        Convert a binary mask to run-length encoding.

        Parameters:
            binary_mask (numpy.ndarray): A 2D array where 1s represent the mask and 0s the background.

        Returns:
            dict: A dictionary with two keys: 'counts' and 'size', where 'counts' holds the RLE and 'size' the dimensions of the mask.
        '''
        # Initialize the RLE dictionary with the size of the mask
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        
        # Prepare to fill the counts list in the RLE dictionary
        counts = rle['counts']
        
        # Flatten the array column-wise and group by value
        flattened = binary_mask.ravel(order='F')
        grouped = itertools.groupby(flattened)
        
        # Iterate through grouped data to form the RLE
        for i, (pixel_value, group_iter) in enumerate(grouped):
            # Convert group iterator to list to count elements
            group_length = len(list(group_iter))
            
            # If first group is mask, prepend 0
            if i == 0 and pixel_value == 1:
                counts.append(0)
            
            # Append the length of each group to the RLE counts
            counts.append(group_length)
        
        return rle

    
    def create_annotations(self, df, images):
        annotations = []
        count = 0 

        for image in tqdm(images, desc='Generating annotations'):
            image_id = image['id']
            filepath = image['file_name']
            file_id = ('_'.join(
                (filepath.split("/")[-3] + "_" + filepath.split("/")[-1]).split("_")[:-4]))
            height_slice = int(filepath.split("/")[-1].split("_")[3])
            width_slice = int(filepath.split("/")[-1].split("_")[2])
        
            ids = df.index[df['id'] == file_id].tolist()
            if len(ids) > 0:
                for idx in ids:
                    segmentation_mask = self.decode_rle(
                        df.iloc[idx]['segmentation'], (height_slice, width_slice))
                    for contour in cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]:
                        mask_image = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                        cv2.drawContours(mask_image, [contour], -1, 255, -1)
                        encoded_segmentation = self.convert_binary_mask_to_rle(mask_image)
                        ys, xs = np.where(mask_image)
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        annotations.append({
                            'segmentation': encoded_segmentation,
                            'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # (x, y, w, h) format
                            'area': mask_image.sum(),
                            'image_id': image_id,
                            'category_id': CLASSES.index(df.iloc[idx]['class']),
                            'iscrowd': 0,
                            'id': count
                        })
                        count += 1
        return annotations

    
class DataGenerator(Dataset):
  # function getting info dataset from json coco
  # Batch size
  # subset train or test for annotations
  # image_list to develop... 
  # classes classe wanted
  # input image size tuple (X,X)
  # annFile path to annoted coco json file file 
    def __init__(self, dataset_dir, batch_size, subset, classes, 
                 input_image_size, annFile, shuffle=False):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.batch_size = batch_size
        self.classes= classes
        self.coco = COCO(annFile)
        self.catIds = self.coco.getCatIds(catNms=self.classes)
        self.cats = self.coco.loadCats(self.catIds)
        self.imgIds = self.coco.getImgIds()
        self.image_list = self.coco.loadImgs(self.imgIds)
        self.indexes = np.arange(len(self.image_list))
        self.input_image_size= (input_image_size)
        self.dataset_size = len(self.image_list)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
      return int(len(self.image_list)/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)      

    def get_class_name(self, class_id, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == class_id:
                return cats[i]['name']
        return None
  
    def get_normal_mask(self, image_id, catIds):
        annIds = self.coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros(self.input_image_size, dtype=np.uint8)
        for a in range(len(anns)):
            className = self.get_class_name(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className)+1
            new_mask = cv2.resize(self.coco.annToMask(anns[a])*pixel_value, self.input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        return train_mask          
        

    def get_levels_mask(self, image_id):
      #for each category , we get the x mask and add it to mask list
      masks = []  
      mask = np.zeros((self.input_image_size))
      for catId in self.catIds:
        mask = self.get_normal_mask(image_id, catId)
        masks.append(mask)
      return masks       

    def get_image(self, file_path):
        train_img = cv2.imread(os.path.join(self.dataset_dir, file_path), cv2.IMREAD_ANYDEPTH)
        train_img = cv2.resize(train_img, (self.input_image_size))
        train_img = train_img.astype(np.float32) / 255.
        if (len(train_img.shape)==3 and train_img.shape[2]==3): 
            return train_img
        else: 
            stacked_img = np.stack((train_img,)*3, axis=-1)
            return stacked_img          
    
    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3))
        y = np.empty((self.batch_size, 128, 128, 3))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i in range(len(indexes)):
          value = indexes[i]
        #   print(f"Index value: {value}")
        #   print(f"Content at index: {self.image_list[value]}")
          img_info = self.image_list[value]
          w = img_info['height']
          h = img_info['width']

          X[i,] = self.get_image(img_info['file_name']) 
          mask_train = self.get_levels_mask(img_info['id'])
          for j in self.catIds:
            y[i,:,:,j] = mask_train[j]   

        X = np.array(X)
        y = np.array(y)

        if self.subset == 'train':
            return X, y
        else: 
            return X


        

