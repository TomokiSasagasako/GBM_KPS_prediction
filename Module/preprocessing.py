# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import os
import glob
import math
from decimal import Decimal, ROUND_HALF_UP
import cv2
import nibabel as nib
from scipy import interpolate, stats
import numpy as np
from Module.extractor import Extractor
from Module.model import Unet_model

load_model_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'segmentation.hdf5')

def get_num_layers(nii_img):
    zaxis = nii_img.header['pixdim'][3]
    slice_num = nii_img.get_fdata().shape[2] * zaxis / 4.0
    slice_num = int(Decimal(str(slice_num)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    return slice_num

def reshape_image(original_img_path, process_img_path, min_num_layers):
    ''''
    input:
      original_img - .nii or .nii.gz image path which are fitted to.
      process_img - .nii or .nii.gz image path to be resized.
    output:
      resized_array - ndarray
    '''
    original_img = nib.load(original_img_path)
    process_img = nib.load(process_img_path)
    original_shape = original_img.get_fdata().shape
    resize = original_img.header['pixdim'][1]/process_img.header['pixdim'][1]
    img0 = process_img.get_fdata()
    export_img_size = max(original_shape[0], original_shape[1])
    export_img=np.zeros((export_img_size, export_img_size, get_num_layers(process_img)),dtype=np.int16)
    if len(img0.shape)==4:
        img0=img0[:,:,:,-1]
    img1=np.zeros((int(img0.shape[0]/resize),int(img0.shape[1]/resize), img0.shape[2]),dtype=np.int16)
    x0=range(0,img0.shape[0])
    y0=range(0,img0.shape[1])
    x1=np.linspace(0,img0.shape[0]-1,img1.shape[0])
    y1=np.linspace(0,img0.shape[1]-1,img1.shape[1])
    for i in range(0,img0.shape[2]):
        f=interpolate.interp2d(x0,y0,img0[:,:,i].T,kind='linear')
        img1[:,:,i]=f(x1,y1).T
    if img1.shape[2] != export_img.shape[2]:
        img2=np.zeros((img1.shape[0],img1.shape[1], export_img.shape[2]),dtype=np.int16)
        z0=range(0,img0.shape[2])
        z1=np.linspace(0,img0.shape[2]-1,img2.shape[2])
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                fn = interpolate.interp1d(z0, img1[i,j,:])
                img2[i,j,:] = fn(z1)
    else: 
        img2 = img1
    if (export_img.shape[0] >= img2.shape[0]) & (export_img.shape[1] >= img2.shape[1]):
        export_img[(export_img.shape[0]-img2.shape[0])//2:(export_img.shape[0]+img2.shape[0])//2,
                (export_img.shape[1]-img2.shape[1])//2:(export_img.shape[1]+img2.shape[1])//2, :]=img2
    elif (export_img.shape[0] < img2.shape[0]) & (export_img.shape[1] < img2.shape[1]):
        export_img = img2[(img2.shape[0]-export_img.shape[0])//2:(img2.shape[0]+export_img.shape[0])//2,
        (img2.shape[1]-export_img.shape[1])//2:(img2.shape[1]+export_img.shape[1])//2, :]
    else: print("特殊パターンのため相談ください - 対象ファイルパス: %s" % process_img_path)
    start_ind = math.floor((export_img.shape[2]-min_num_layers)/2)
    export_img = export_img[:,:,start_ind:start_ind+min_num_layers]
    return export_img


def create_mask(t1_img_array, dwi_img_array, mask_threshold = 0.5, dwi_threshold = 0.1):
    extractor = Extractor()
    mask = (extractor.run(t1_img_array) > mask_threshold)|(dwi_img_array > np.percentile(dwi_img_array, (1-dwi_threshold)*100))
    mask_img = (1 * mask).astype(np.uint8)
    return mask, mask_img


def apply_mask(img_array, mask):
    img_array[~mask] = 0
    return img_array


class Prediction(object):
    def __init__(self, batch_size):
        self.batch_size=batch_size
        unet=Unet_model(img_shape=(240,240,4), load_model_weights=load_model_path)
        self.model=unet.model
        self.extractor = Extractor()

    def predict_volume(self, filepath_image, mask_threshold=0.5, removal_threshold=0.1):
        '''
        segment the input volume
        INPUT   (1) str 'filepath_image': filepath of the volume to predict
                (2) bool 'show': True to ,
        OUTPUT  (1) np array of the predicted volume
                (2) np array of the flair image
        '''
        #read datasets
        checklists = set(['flair', 't1', 't1ce', 'adc', 'dwi'])
        filelists = set([img.split('/')[-1].split('.')[0] for img in glob.glob(filepath_image + '/*.nii')])
        assert checklists<=filelists, '以下の必要ファイルがありません：%s'% list(checklists ^ filelists)
        flair= filepath_image + '/flair.nii'
        t1s = filepath_image + '/t1.nii'
        t1c = filepath_image + '/t1ce.nii'
        adc = filepath_image + '/adc.nii'
        dwi = filepath_image + '/dwi.nii'
        path_lists = [flair, t1s, t1c, adc, dwi]
        slice_nums = self._unify_layers(path_lists)
        pixel_size = nib.load(t1s).header['pixdim'][1]
        array_lists = [reshape_image(t1s, path, slice_nums) for path in path_lists]
        # export original images
        original_images = list()
        for i in range(len(array_lists)):
            original_images.append(array_lists[i])
        original_images = np.transpose(original_images,(3,1,2,0))
        # process and export masked images
        mask, mask_img = create_mask(array_lists[1], array_lists[4], mask_threshold, removal_threshold)
        processed_images = list()
        for img_array in array_lists[:4]:
            img_array = apply_mask(img_array, mask)
            resized_slices = np.zeros((240, 240, img_array.shape[2]))
            for slice_i in range(img_array.shape[2]):
                resized_slices[:,:,slice_i] = cv2.resize(self._normalize(img_array[:,:,slice_i]), (240, 240), interpolation=cv2.INTER_NEAREST)
            processed_images.append(resized_slices)
        processed_images = np.transpose(processed_images,(3,1,2,0))
        slices, h, w, modal = processed_images.shape
        output = np.ones((slices, h, w), dtype=np.uint8)*4
        # predict classes of each pixel based on the model
        prediction=self.model.predict(processed_images, batch_size=self.batch_size, verbose=0)
        prediction=np.argmax(prediction, axis=-1)
        prediction=prediction.astype(np.uint8)
        # export mask images
        resized_mask = np.zeros((240, 240, mask_img.shape[2]))
        for slice_i in range(mask_img.shape[2]):
              resized_mask[:,:,slice_i] = cv2.resize(mask_img[:,:,slice_i], (240, 240), interpolation=cv2.INTER_NEAREST)
        resized_mask = np.transpose(resized_mask, (2,0,1))
        output[prediction==1]=1
        output[prediction==2]=2
        output[prediction==3]=3
        output[resized_mask<mask_threshold]=0
        output[(processed_images[:,:,:,0]<stats.norm.ppf(removal_threshold))&(processed_images[:,:,:,1]<stats.norm.ppf(removal_threshold))]=0
        del mask_img, resized_mask, processed_images
        return slice_nums, pixel_size, np.array(original_images), output.astype(np.uint8)

    def _normalize(self, slice_img):
        b = np.percentile(slice_img, 99)
        t = np.percentile(slice_img, 1)
        slice_img = np.clip(slice_img, t, b)
        image_nonzero = slice_img[np.nonzero(slice_img)]
        if np.std(slice_img)==0 or np.std(image_nonzero) == 0:
            return slice_img
        else:
            tmp= (slice_img - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp==tmp.min()]=-9
            return tmp
            
    def _unify_layers(self, path_lists):
        slice_nums = list()
        for path in path_lists:
            nii_img = nib.load(path) 
            slice_nums.append(get_num_layers(nii_img))
        return min(slice_nums)