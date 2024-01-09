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
import warnings
import datetime
import pydicom
import dicom2nifti
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Specify directory of DICOM files')
parser.add_argument('--working_dir', type=str, help='Specify working directory')
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.working_dir

def dicom_to_nifti(study_id, dicom_dir):
    try:
        ds = pydicom.dcmread(glob.glob(dicom_dir + '/*')[0], force=True)
        study_date = ds.StudyDate
        series = ds.SeriesDescription
        output_dir = OUTPUT_DIR + f'/{study_id}/{study_date}'
        os.makedirs(output_dir, exist_ok=True)
        if ('t1' in series) & ~('ce' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('t1' in series) & ('ce' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('t2' in series) & ('tse' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t2.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t2', reorient_nifti=True)
        elif ('t2' in series) & ('flair' in series) & ('3mm' in series) & ('axi' in series) & ~('MPR' in series) & ~(os.path.exists(output_dir+'/flair.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True)
        elif ('DWI' in series) & ~('ADC' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('DWI' in series) & ('ADC' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/adc.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        else: pass
    except: 
        study_date, series = None, None
    return study_date, series


def dicom_to_nifti_debug(study_id, dicom_dir):
    try:
        ds = pydicom.dcmread(glob.glob(dicom_dir + '/*')[0], force=True)
        study_date = ds.StudyDate
        series = ds.SeriesDescription
        output_dir = OUTPUT_DIR + f'/{study_id}/{study_date}'
        #The following are search codes for individual cases
        if ('t1' in series) & ~('ce' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t1.nii')):  ##T1W
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('T1w' in series) & ~('ce' in series) & ('3D' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('T1w' in series) & ~('ce' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('t1' in series) & ~('ce' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('T1' in series) & ~('ce' in series) & ('SAG' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('mpr' in series) & ~('ce' in series) & ('axi' in series) & ('3mm' in series) & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('T1' in series) & ~('ce' in series) & ('Ax' in series)  & ~(os.path.exists(output_dir+'/t1.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1', reorient_nifti=True)
        elif ('t1' in series) & ('ce' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):  ##T1CE
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)        
        elif ('T1w' in series) & ('ce' in series) & ('3D' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('T1w' in series) & ('ce' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('t1' in series) & ('ce' in series) & ('sag' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('T1' in series) & ('CE' in series) & ('SAG' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('T1' in series) & ('CE' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('T1' in series) & ('CE' in series) & ('Ax' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('mpr' in series) & ('ce' in series) & ('axi' in series) & ('3mm' in series) & ~(os.path.exists(output_dir+'/t1ce.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t1ce', reorient_nifti=True)
        elif ('t2' in series) & ~('flair' in series) & ('tse' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/t2.nii')):  ##T2W
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t2', reorient_nifti=True)
        elif ('t2' in series) & ~('flair' in series) & ('Ax' in series) & ~(os.path.exists(output_dir+'/t2.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t2', reorient_nifti=True)
        elif ('T2' in series) & ~('flair' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/t2.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t2', reorient_nifti=True)
        elif ('T2' in series) & ~('FLAIR' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/t2.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/t2', reorient_nifti=True)
        elif ('flair' in series) & ('3mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/flair.nii')):  ##FLAIR
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True) 
        elif ('Flair' in series) & ('Ax' in series) & ~(os.path.exists(output_dir+'/flair.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True)    
        elif ('flair' in series) & ('2mm' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/flair.nii')):  
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True) 
        elif ('T2' in series) & ('FLAIR' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/flair.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True)
        elif ('FLAIR' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/flair.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True)
        elif ('FLAIR' in series) & ('Ax' in series) & ~(os.path.exists(output_dir+'/flair.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/flair', reorient_nifti=True)
        elif ('DWI' in series) & ~('ADC' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):  ##DWI
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('DWI' in series) & ~('ADC' in series) & ('Ax' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('ep2d' in series) & ~('ADC' in series) & ('axi' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('DWI' in series) & ~('ADC' in series) & ('AXI' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('DWI' in series) & ~('ADC' in series) & ('3mm' in series) & ~(os.path.exists(output_dir+'/dwi.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/dwi', reorient_nifti=True)
        elif ('DWI' in series) & ('ADC' in series)  & ('axi' in series) & ~(os.path.exists(output_dir+'/adc.nii')):  ##ADC
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        elif ('DWI' in series) & ('ADC' in series)  & ('Ax' in series) & ~(os.path.exists(output_dir+'/adc.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        elif ('DWI' in series) & ('ADC' in series)  & ('AXI' in series) & ~(os.path.exists(output_dir+'/adc.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        elif ('ep2d' in series) & ('ADC' in series)  & ('axi' in series) & ~(os.path.exists(output_dir+'/adc.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        elif ('DWI' in series) & ('ADC' in series)  & ('3mm' in series) & ~(os.path.exists(output_dir+'/adc.nii')):
            dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir + '/adc', reorient_nifti=True)
        else: pass
    except: 
        study_date, series = None, None
    return study_date, series


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


## 実行コード
if __name__ == '__main__':
    t = Time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patient_dirs = sorted([path for path in glob.glob(INPUT_DIR + '/BT*') if os.path.isdir(path)])
    with tqdm(total=len(patient_dirs), desc='[Covert DICOM to Nifti]') as pbar:
        for patient_dir in patient_dirs:
            study_id = patient_dir.split('/')[-1].split('_')[0]
            debug_dates_lists, debug_series_lists = list(), list()
            dicom_dirs = [path for path in glob.glob(patient_dir + '/*/*') if os.path.isdir(path)]
            for dicom_dir in dicom_dirs:
                study_date, series = dicom_to_nifti(study_id, dicom_dir)
            for dicom_dir in dicom_dirs:
                study_date, series = dicom_to_nifti_debug(study_id, dicom_dir)
                if series is not None:
                    debug_dates_lists.append(study_date)
                    debug_series_lists.append(series)            
            debug_data = pd.Series(debug_dates_lists, index = debug_series_lists)
            debug_dates = debug_data.unique().tolist()
            for study_date in debug_dates:
                checklists = set(['flair', 't1', 't1ce', 'adc', 'dwi'])
                filelists = set([img.split('/')[-1].split('.')[0] for img in glob.glob(OUTPUT_DIR + f'/{study_id}/{study_date}/*.nii')])
                if not (checklists <= filelists):
                    print('症例ID = %s_GBM, 画像撮影日 = %s'%(str(study_id), str(study_date)))
                    diff_list = list(checklists - filelists)
                    for name in diff_list:
                        print('変換ファイルに必要な画像を認めません:%s'%name)
                    print(debug_data[debug_data==study_date].index)
            pbar.update(1)
    print("Time to completed tasks : %s" % str(t.elapsed()))