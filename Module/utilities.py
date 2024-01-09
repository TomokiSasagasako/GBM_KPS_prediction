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
import math
import cv2
import datetime
from scipy import integrate
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def color_img(predicted_image):
    """
    input: (1) predicted_image: np.array(height, width)
          (2) flair_image: np.array(height, width)
    output: synthesized (color-mapped) image: np.array(height, width, channel)
    """
    h, w = predicted_image.shape
    mat = np.zeros((3, h, w))
    # red region
    mat[0][predicted_image==1] = 255
    # green region
    mat[1][predicted_image==2] = 255
    # yellow region (tumor lesion enhanced with contrast material)
    mat[0][predicted_image==3] = 255
    mat[1][predicted_image==3] = 255
    # mask region
    mat[0][predicted_image==4] = 128
    mat[1][predicted_image==4] = 128
    mat[2][predicted_image==4] = 128
    output = np.transpose(mat, (1, 2, 0)).astype(np.uint8)
    return output


# convert img(240, 240, 3) into array(240, 240)
def img2array(img):
    h, w, _ = img.shape
    mat = np.zeros((h, w))
    # red region
    mat[img[:,:,0]==255] = 1
    # green region
    mat[img[:,:,1]==255] = 2
    # yellow region
    mat[(img[:,:,0]==255)&(img[:,:,1]==255)] = 3
    # gray region
    mat[(img[:,:,0]==128)&(img[:,:,1]==128)&(img[:,:,2]==128)] = 4
    return mat


def plot_predicted_images(predicted_images, start_ind=0, save_path=None):
    row = math.ceil(predicted_images.shape[0] / 5)
    fig = plt.figure(figsize=(8, row*2))
    for i in range(predicted_images.shape[0]):
        plt.subplot(row, 5, i+1)
        plt.imshow(color_img(predicted_images[i]), origin='lower')
        plt.gca().set_title('Slice_' + str(i+start_ind+1), fontsize=12)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close(fig)


def plot_original_images(original_images, save_path):
    images = ['FLAIR', 'T1-weighted', 'T1-enhanced', 'ADC map', 'DWI']
    for im in range(len(images)):
        image_array = original_images[:, :, :, im]
        row = math.ceil(image_array.shape[0] / 5)
        fig = plt.figure(figsize=(8, row*2))
        for i in range(image_array.shape[0]):
            plt.subplot(row, 5, i + 1)
            plt.imshow(image_array[i, :, :], cmap='gray', origin='lower')
            plt.gca().set_title('Slice_' + str(i + 1), fontsize=12)
            plt.gca().xaxis.set_visible(False)
            plt.gca().yaxis.set_visible(False)
        plt.savefig(save_path + 'plot_%s.jpg' % images[im])
        plt.clf()
        plt.close(fig)


def plot_predicted_slice(original_images, predicted_images, slice_num, save_path=None):
    fig = plt.figure(figsize=(28, 4))
    images = ['FLAIR', 'T1-weighted', 'T1-enhanced', 'ADC map', 'DWI']
    for im in range(len(images)):
        plt.subplot(1, 6, im+1)
        plt.imshow(original_images[slice_num,:,:,im], cmap='gray', origin='lower')
        plt.gca().set_title(images[im] + ' image')
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
    plt.subplot(1, 6, 6)
    plt.imshow(color_img(predicted_images[slice_num]), origin='lower')
    plt.gca().set_title('Predicted image')
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close(fig)


def calculate_area_diameter(predicted_img, threthold, fill_tumor=False):
    ret, bin_img = cv2.threshold(predicted_img, 0, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(bin_img)
    try:
        if fill_tumor:
            try:
                kernel = np.ones((20,20), np.uint8)
                countours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_img = cv2.drawContours(bin_img, countours, -1, color=255, thickness=20)
                countours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_img = cv2.drawContours(mask_img, countours,-1, color=255, thickness=-1)
                erosion = cv2.erode(mask_img, kernel, iterations=-1)
                mask[erosion==255] = 255
            except: pass
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Select the contour with the largest area
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
        if cv2.contourArea(max_cnt) > threthold:
            # Draw a black background image with only the largest outline filled in
            cv2.drawContours(mask, [max_cnt], -1, color=255, thickness=-1)
            predicted_img[mask == 0] = 0
            max_diameter, orth_diameter = calculate_diameter(max_cnt)
            area = cv2.contourArea(max_cnt)
        else:
            predicted_img = np.zeros_like(bin_img)
            area, max_diameter, orth_diameter = 0, 0, 0
    except:
        predicted_img = np.zeros_like(bin_img)
        area, max_diameter, orth_diameter = 0, 0, 0
    return max_diameter, orth_diameter, area, predicted_img


def calculate_all_area_diameter(predicted_images, threthold, fill_tumor=False):
    predicted_max_diameter = list()
    predicted_orth_diameter = list()
    predicted_areas = list()
    predicted_imgs = list()
    for i in range(predicted_images.shape[0]):
        max_diameter, orth_diameter, area, predicted_img = calculate_area_diameter(predicted_images[i],threthold=threthold)
        predicted_max_diameter.append(max_diameter)
        predicted_orth_diameter.append(orth_diameter)
        predicted_areas.append(area)
        predicted_imgs.append(predicted_img)
    return np.array(predicted_max_diameter), np.array(predicted_orth_diameter), np.array(predicted_areas), np.array(predicted_imgs)


def calculate_diameter(counter, threshold=0.01):
    max_diameter, orth_diameter = 0, 0
    for p1, p2 in combinations(counter[:, 0], 2):
        x1, y1 = p1
        x2, y2 = p2
        dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dist > max_diameter:
            max_diameter = dist
            x_dist, y_dist = x2 - x1, y2 - y1
    for p1, p2 in combinations(counter[:, 0], 2):
        a1, b1 = p1
        a2, b2 = p2
        # Calculate the absolute cosine values from the inner products,
        # consider it orthogonal if the less than threthold.
        dist = (a2 - a1) ** 2 + (b2 - b1) ** 2
        ip = (a2 - a1) * x_dist + (b2 - b1) * y_dist
        if dist != 0:
            cos = ip / (math.sqrt(max_diameter) * math.sqrt(dist))
            if (abs(cos) < threshold) & (dist > orth_diameter):
                orth_diameter = dist
    return math.sqrt(max_diameter), math.sqrt(orth_diameter)


def area_lists(pixel_size, predicted_images, threthold):
    """
    pixel_size: pixel size of original images
    predicted_images: image arrays after segmentation predictions
    """
    predicted_images[predicted_images == 4] = 0
    _, _, areas_total, predicted_images = calculate_all_area_diameter(predicted_images, threthold=threthold)
    predicted_images[predicted_images == 2] = 0
    _, _, areas_red_yellow, _ = calculate_all_area_diameter(predicted_images, threthold=0)
    predicted_images[predicted_images == 1] = 0
    max_diameters, orth_diameters, areas_yellow, _ = calculate_all_area_diameter(predicted_images, threthold=0, fill_tumor=True)
    max_diameter = max(max_diameters) * (pixel_size ** 2)
    orth_diameter = max(orth_diameters) * (pixel_size ** 2)
    areas_total = areas_total * (pixel_size ** 2) / 100
    areas_red_yellow = areas_red_yellow * (pixel_size ** 2) / 100
    areas_yellow = areas_yellow * (pixel_size ** 2) / 100
    return max_diameter, orth_diameter, areas_total, areas_red_yellow, areas_yellow


def calculate_volume(array_layers, array_areas):
    # The x-axis is calculated using layer_num * 0.4, since it is calculated in 4mm increments.
    x = array_layers * 0.4
    y = array_areas
    estimated_volume = integrate.simps(y, x)
    return estimated_volume


def detect_clipping_area(area_arrays, exclusion, threthold=75):
    top_lists = ((area_arrays > np.percentile(area_arrays, threthold)) * 1).tolist()
    area_group_dicts = dict()
    group_count = 0
    for idx in range(len(top_lists)):
        if idx < exclusion: 
            area_group_dicts.update({idx: 0})
        elif idx >= (len(top_lists) - exclusion):
            area_group_dicts.update({idx: 0})
        elif (top_lists[idx-1] == 0) & (top_lists[idx] == 1):
            group_count += 1
            area_group_dicts.update({idx: group_count})
        elif (top_lists[idx-1] == 1) & (top_lists[idx] == 1):
            area_group_dicts.update({idx: group_count})
        else:
            area_group_dicts.update({idx: 0})
    ds_area_group = pd.Series(area_group_dicts)
    ds_area_group = ds_area_group[ds_area_group != 0]
    max_group_num = ds_area_group.value_counts().index[0]
    index_max_group = np.array(ds_area_group[ds_area_group == max_group_num].index).astype(int)
    index_median = int(np.percentile(index_max_group, 50))
    if index_median < (12 + exclusion -1):
        start_ind = exclusion
    else:
        start_ind = min((index_median - 12 + 1), (len(top_lists) - 24 - exclusion))
    return start_ind


def remove_small_objects(array, threshold=300):
    _, bin_img = cv2.threshold(array, 0, 1, cv2.THRESH_BINARY)
    mask = np.zeros_like(bin_img)
    try:
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Select the contour with the largest area
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
        if cv2.contourArea(max_cnt) > threshold:
        # Fill and draw the largest outline
            cv2.drawContours(mask, [max_cnt], -1, color=1, thickness=-1)
            array[mask==0] = 0
            area = cv2.contourArea(max_cnt)
        else: 
            array = np.zeros_like(bin_img)
            area = 0
    except: 
        array = np.zeros_like(bin_img)
        area = 0
    return area, array


def ellipse_tumor(array):
    mat = np.zeros_like(array)
    mat[(array==1)|(array==3)] = 1
    _, bin_img = cv2.threshold(mat, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) >0:
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
        if len(max_cnt) >= 6:
            ellipse = cv2.fitEllipse(max_cnt)
            if ('nan' in ellipse[0]) or ('inf' in ellipse[0]) or ('na' in ellipse[1]) or ('inf' in ellipse[1]):
                return None
            return ellipse
        else: 
            return None
    else:
        return None


def find_nearest_edge(array, start, kernel_factor=5):
    mask = np.zeros_like(array)
    kernel = np.ones((kernel_factor, kernel_factor), np.uint8)
    _, bin_img = cv2.threshold(array, 0, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_img = cv2.drawContours(bin_img, contours, -1, color=1, thickness=kernel_factor)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_img = cv2.drawContours(bin_img, contours, -1,  color=1, thickness=-1)
    erosion = cv2.erode(mask_img, kernel, iterations=-1)
    mask[erosion==1] = 1
    _, bin_img = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    dist = array.shape[0] ** 2 + array.shape[1] ** 2
    for p in max_cnt:
        if 100<=p[0][1]<=140 and 100<p[0][0]<240: continue
        d = (p[0][0]-start[0])**2 + (p[0][1]-start[1])**2
        if d < dist:
            dist = d
            point = p
    return point


def predict_resection_slice(array, edge_rate=0.6, elp_rate=1.0):
    _, array = remove_small_objects(array)
    # Calculate the ellipse circumscribed to the tumor
    elp_tumor = ellipse_tumor(array)
    if elp_tumor is not None:
        elp_tumor = (elp_tumor[0], (elp_tumor[1][0]*elp_rate, elp_tumor[1][1]*elp_rate), elp_tumor[-1])
        center = find_nearest_edge(array, elp_tumor[0])[0]
        axis_1 = (((elp_tumor[0][0]-center[0])**2+(elp_tumor[0][1]-center[1])**2)**0.5)*2
        axis_2 = min(elp_tumor[1])*edge_rate
        angle = np.degrees(np.arctan2(center[1]-elp_tumor[0][1], center[0]-elp_tumor[0][0]))
        img = color_img(array)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.ellipse(img_bgr, elp_tumor, (0,0,0), thickness=-1)
        cv2.ellipse(img_bgr, ((int(center[0]), int(center[1])), (int(axis_1), int(axis_2)), int(angle)), (0,0,0), thickness=-1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = color_img(array)
    return img_rgb


# draw and save the image and array for each patient
def predict_resection_patient(pre_arrays, save_dir, edge_rate=0.6, elp_rate=1.0):
    resection_arrays = []
    row = math.ceil(pre_arrays.shape[0]/5)
    fig = plt.figure(figsize=(8, row*2))
    for i, array in enumerate(pre_arrays):
        res_img = predict_resection_slice(array, edge_rate=edge_rate, elp_rate=elp_rate)
        res_array = img2array(res_img)
        resection_arrays.append(res_array)
        plt.subplot(row, 5, i+1)
        plt.imshow(res_img)
        plt.gca().set_title('Slice_'+str(i+1), fontsize=12)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
    resection_arrays = np.array(resection_arrays)
    if save_dir is not None:
        np.save(save_dir+'/array/predicted_resection_array', resection_arrays)
        plt.savefig(save_dir + '/predicted_resection_imgs.jpg')
    plt.show()
    plt.clf()
    plt.close(fig)
    return resection_arrays


class Time:
    """
    Class for displaying elapsed time.
    """
    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed
