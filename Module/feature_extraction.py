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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Module.utilities import *
from Module.preprocessing import *
from Module.vae_3class import lesion_feature_extractions
from Module.vae_mask import mask_feature_extractions
import silence_tensorflow.auto
import tensorflow as tf
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Specify directory of datasets files')
parser.add_argument('--working_dir', type=str, help='Specify working directory')
parser.add_argument('--contour_threshold', type=int, default=300, help='Set the minimum threthold of area for contouring')
parser.add_argument('--uniform_clip', default=False, help='If true, clipping slices from the same starting points', action='store_true')
parser.add_argument('--exclusion', type=int, default=3, help='The number of slices to exclude from the beginning and end of slices in the analysis')
parser.add_argument('--batch_size', type=int, default=4, help='Specify batch size')
parser.add_argument('-dim_lesion', '--latent_dim_lesion', type=int, default=24, help='Specify latent dimensions of feature extractions from lesions')
parser.add_argument('-dim_mask', '--latent_dim_mask', type=int, default=16, help='Specify latent dimensions of feature extractions from masks')
parser.add_argument('-s', '--save', default=False, help='Save original and predicted images', action='store_true')
parser.add_argument('--kernel', type=int, default=5, help='Dilation and kernel size to detect contrast-enhanced area')
parser.add_argument('--predict_resection', default=False, help='Predict resected area from pre-predicted images', action='store_true')
parser.add_argument('--edge_rate', type=float, default=0.6, help='Set the outer resection area adjacent to the resection ellipse of the tumor')
parser.add_argument('--elp_rate', type=float, default=1.0, help='Set the ratio of the ellipse circumscribed by the tumor')

args = parser.parse_args()

INPUT_DIR =  args.input_dir
OUTPUT_DIR = args.working_dir
np.random.seed(0)


def plot_patient(patient_dir_path, save=False, exclusion=args.exclusion):
    """
    input: directory path of the patient datasets
    """
    paths = sorted(glob.glob(os.path.join(patient_dir_path, '*')))
    dirs = [path for path in paths if os.path.isdir(path)]
    assert len(dirs)==2, '術前後の画像が揃っていません'
    pre_path = dirs[0]
    post_path = dirs[1]
    seg_model = Prediction(batch_size=2)
    pre_slice_nums, pre_pixel_size, pre_original_images, pre_predicted_images = seg_model.predict_volume(pre_path)
    post_slice_nums, post_pixel_size, post_original_images, post_predicted_images = seg_model.predict_volume(post_path)
    
    if pre_slice_nums>post_slice_nums:
        pre_original_images = pre_original_images[(pre_slice_nums-post_slice_nums)//2:(pre_slice_nums-post_slice_nums)//2+post_slice_nums,:,:,:]
        pre_predicted_images = pre_predicted_images[(pre_slice_nums-post_slice_nums)//2:(pre_slice_nums-post_slice_nums)//2+post_slice_nums,:,:]
    if pre_slice_nums<post_slice_nums:
        post_original_images = post_original_images[(post_slice_nums-pre_slice_nums)//2:(post_slice_nums-pre_slice_nums)//2+pre_slice_nums,:,:,:]
        post_predicted_images = post_predicted_images[(post_slice_nums-pre_slice_nums)//2:(post_slice_nums-pre_slice_nums)//2+pre_slice_nums,:,:]
        
    # Get patient data acquisition date
    patient = patient_dir_path.split('/')[-1]
    pre_date = pre_path.split('/')[-1]
    post_date = post_path.split('/')[-1]
    save_dir = os.path.join(args.working_dir, patient)
    os.makedirs(save_dir, exist_ok=True)

    if save:
        os.makedirs(save_dir + '/pre', exist_ok=True)
        os.makedirs(save_dir + '/post', exist_ok=True)
        # save preoperative images
        for slice_num in range(len(pre_predicted_images)):
            plot_predicted_slice(pre_original_images, pre_predicted_images, slice_num,
                                 save_path=save_dir + '/pre/pre_slice_%s.jpg' % str(slice_num))
        plot_predicted_images(pre_predicted_images, save_path=save_dir + '/pre/pre_plot_predictions.jpg')
        plot_original_images(pre_original_images, save_path=save_dir + '/pre/pre_')
        # save postoperative images
        for slice_num in range(len(post_predicted_images)):
            plot_predicted_slice(post_original_images, post_predicted_images, slice_num,
                                 save_path=save_dir + '/post/post_slice_%s.jpg' % str(slice_num))
        plot_predicted_images(post_predicted_images, save_path=save_dir + '/post/post_plot_predictions.jpg')
        plot_original_images(post_original_images, save_path=save_dir + '/post/post_')

    # Calculate diameter and tumor areas of predictions images        
    pre_max_diameter, pre_orth_diameter, pre_total, pre_red_yellow, pre_yellow = area_lists(pre_pixel_size, pre_predicted_images.copy(), threthold=args.contour_threshold)
    _, _, post_total, post_red_yellow, post_yellow = area_lists(post_pixel_size, post_predicted_images.copy(), threthold=args.contour_threshold)

    num_slices = pre_predicted_images.shape[0]
    if args.uniform_clip:
        start = num_slices-24-exclusion
    else:
        start = detect_clipping_area(pre_total, exclusion)

    slices = np.arange(24)
    pre_predicted_images = pre_predicted_images[start:start+24]
    pre_total, pre_red_yellow, pre_yellow = pre_total[start:start+24], pre_red_yellow[start:start+24], pre_yellow[start:start+24]
    pre_volume = calculate_volume(slices, pre_yellow)
    post_predicted_images = post_predicted_images[start:start+24]
    post_total, post_red_yellow, post_yellow = post_total[start:start+24], post_red_yellow[start:start+24], post_yellow[start:start+24]
    post_volume = calculate_volume(slices, post_yellow)
    slices = slices + start + 1

    # Export pre/post image array
    if save:
        os.makedirs(save_dir + '/array', exist_ok=True)
        np.save(save_dir + '/array/pre_predicted_array', pre_predicted_images)
        np.save(save_dir + '/array/post_predicted_array', post_predicted_images)

    # Export results as dataframe
    results = dict()
    results.update({'patient_id': patient,
                    'preoperative_mri_date': pre_date,
                    'postoperative_mri_date': post_date,
                    'estimated_preoperative_tumor_volume': pre_volume,
                    'estimated_maximum_tumor_diameter1': pre_max_diameter,
                    'estimated_maximum_tumor_diameter2': pre_orth_diameter,
                    'estimated_postoperative_tumor_volume': post_volume,
                    'start_slice_num': start,
                    'end_slice_num': (start + 24 - 1),
                    })
    for i in range(len(pre_red_yellow)):
        results.update({'pre_tumor_dwi_%s' % str(i + 1): pre_red_yellow[i]})
    for i in range(len(post_red_yellow)):
        results.update({'post_tumor_dwi_%s' % str(i + 1): post_red_yellow[i]})
    for i in range(len(pre_red_yellow)):
        results.update({'pre_tumor_%s'%str(i+1): pre_yellow[i]})
    for i in range(len(post_red_yellow)):
        results.update({'post_tumor_%s' % str(i + 1): post_yellow[i]})        
    df_results = pd.DataFrame([results])
    df_results.to_csv(save_dir + '/case_summary.csv', index=False)

    # Plot results
    max_area = max(max(pre_total), max(post_total))
    fig = plt.figure(figsize=(22, 8))
    plt.ion()
    plt.rcParams["font.size"] = 16
    ax1 = fig.add_subplot(121, title=f'Areas of segmentation per slice (Pre-operation:{pre_date})',
                          ylim=(0, max_area + 10), xlabel='slices', ylabel='area (cm2)')
    ax1.bar(slices, pre_yellow, label='tumor (enhanced)', color='red', alpha=0.5)
    ax1.plot(slices, pre_red_yellow, label='tumor (enhanced+DWI low)')
    #ax1.bar(slices, pre_red_yellow, label='tumor (enhanced+DWI low)', color='green', alpha=0.5)
    #ax1.plot(slices, pre_total, label='total')
    ax1.legend()
    ax2 = fig.add_subplot(122, title=f'Areas of segmentation per slice (Post-operation:{post_date})',
                          ylim=(0, max_area + 10), xlabel='slices', ylabel='area (cm2)')
    ax2.bar(slices, post_yellow, label='tumor (enhanced)', color='red', alpha=0.5)
    ax2.plot(slices, post_red_yellow, label='tumor (enhanced+DWI low)')
    #ax2.bar(slices, post_red_yellow, label='tumor (enhanced+DWI low)', color='green', alpha=0.5)
    #ax2.plot(slices, post_total, label='total')
    ax2.legend()
    plt.suptitle(f"Volume comparison in patient: {patient}", fontsize=20)
    plt.savefig(save_dir + '/plot_area_comparison.jpg')
    plt.clf()
    plt.close(fig)
    del pre_original_images, post_original_images
    return df_results.iloc[:, :9], pre_predicted_images, post_predicted_images, start


if __name__ == '__main__':
    t = Time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_summary = pd.DataFrame()
    patient_ids = sorted([path.split('/')[-1] for path in glob.glob(INPUT_DIR + '/*') if os.path.isdir(path)])
    patient_id_lists, pre_images_arrays, post_images_arrays, start_index_lists = list(), list(), list(), list()
    with tqdm(total=len(patient_ids), desc='[Feature extractions from datasets]') as pbar:
        for patient_id in patient_ids:
            patient_dir = os.path.join(INPUT_DIR, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            try:
                df_results, pre_predicted_images, post_predicted_images, start_index = plot_patient(patient_dir, save=args.save)
                df_summary = pd.concat([df_summary, df_results], axis=0)
                pre_images_arrays.append(pre_predicted_images)
            # Predict resection area from pre predicted images
                if args.predict_resection:
                    predicted_resection_array = predict_resection_patient(pre_predicted_images, os.path.join(args.working_dir, patient_id), args.edge_rate, args.elp_rate)
                    post_images_arrays.append(predicted_resection_array)
                else:
                    post_images_arrays.append(post_predicted_images)
                patient_id_lists.append(patient_id)
                start_index_lists.append(start_index)
            except Exception as e:
                print('症例ID %s はデータ欠損のため除外しました' % patient_id)
                print(e)
            pbar.update(1)
    images_arrays = np.concatenate([np.array(pre_images_arrays), np.array(post_images_arrays)])

    # extract features using pretrained vae models  
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    lesion_mean, lesion_recon_img = lesion_feature_extractions(images_arrays, args.batch_size, args.latent_dim_lesion, args.kernel)
    mask_mean, mask_recon_img = mask_feature_extractions(images_arrays, args.batch_size, args.latent_dim_mask)

    # export extracted features to dataframe
    pre_lesion_mean, post_lesion_mean = np.split(lesion_mean, 2)
    pre_mask_mean, post_mask_mean = np.split(mask_mean, 2)
    features = np.concatenate([pre_lesion_mean, pre_mask_mean, post_lesion_mean, post_mask_mean], axis=1)
    pre_lesion_columns = ['pre_lesion_%s' % str(i + 1) for i in range(pre_lesion_mean.shape[1])]
    pre_mask_columns = ['pre_mask_%s' % str(i + 1) for i in range(pre_mask_mean.shape[1])]
    post_lesion_columns = ['post_lesion_%s' % str(i + 1) for i in range(post_lesion_mean.shape[1])]
    post_mask_columns = ['post_mask_%s' % str(i + 1) for i in range(post_mask_mean.shape[1])]
    df_features = pd.DataFrame(data=features, columns=pre_lesion_columns+pre_mask_columns+post_lesion_columns+post_mask_columns)
    df_features['patient_id'] = df_summary['patient_id'].values.tolist()
    df_all_results = pd.merge(df_summary, df_features, on='patient_id')
    df_all_results.to_csv(OUTPUT_DIR+'/features_summary.csv',index=False)

    lesion_recon_img[(lesion_recon_img==0)&(mask_recon_img==1)] = 4
    pre_reconstructed_img, post_reconstructed_img = np.split(lesion_recon_img, 2)
    for idx, patient_id in enumerate(patient_id_lists):
        save_dir = os.path.join(args.working_dir, patient_id)
        plot_predicted_images(pre_reconstructed_img[idx], start_ind=start_index_lists[idx], save_path=save_dir + '/pre_reconstructed_imgs.jpg')
        plot_predicted_images(post_reconstructed_img[idx], start_ind=start_index_lists[idx], save_path=save_dir + '/post_reconstructed_imgs.jpg')

    print("Time to completed tasks : %s" % str(t.elapsed()))
