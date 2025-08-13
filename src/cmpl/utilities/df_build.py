# File created by: Eisa Hedayati
# Date: 10/1/2024
# Description: This file is developed at CMRR

import os
import pandas as pd
import re

def build_medical_data_frame(root_dir):
    def clean_contrast_name(dir_name):
        # Remove 'MR-SE' followed by any digits and a dash from the start of the directory name
        return re.sub(r'^MR-SE\d+-', '', dir_name)

    columns = []
    data = []

    # Ensure root_dir is a valid directory
    if not os.path.isdir(root_dir):
        print(f"Error: The root directory {root_dir} does not exist.")
        return pd.DataFrame()

    # Traverse each study directory
    for study in os.listdir(root_dir):
        study_path = os.path.join(root_dir, study)
        if not os.path.isdir(study_path):
            # print(f"Skipping {study} as it is not a directory.")
            continue
        # print(f"Processing study: {study}")

        # Process Dicoms
        dicoms_path = os.path.join(study_path, 'Dicoms')
        if os.path.isdir(dicoms_path):
            for contrast_dir in os.listdir(dicoms_path):
                contrast_name = clean_contrast_name(contrast_dir)
                dicom_target_path = os.path.relpath(os.path.join(dicoms_path, contrast_dir), root_dir)
                columns.append(('Dicoms', contrast_name, 'Path'))
                data.append((study, dicom_target_path))
        else:
            print(f"No Dicoms directory found in {study_path}")

        # Process h5_files
        h5_files_path = os.path.join(study_path, 'h5_files')
        if os.path.isdir(h5_files_path):
            for file in os.listdir(h5_files_path):
                if file.endswith('.h5'):
                    contrast_name = clean_contrast_name(file.replace('.h5', ''))
                    h5_target_path = os.path.relpath(os.path.join(h5_files_path, file), root_dir)
                    columns.append(('h5_files', contrast_name, 'Path'))
                    data.append((study, h5_target_path))
        else:
            print(f"No h5_files directory found in {study_path}")

        # Process Segmentations
        seg_path = os.path.join(study_path, 'Segmentations')
        if os.path.isdir(seg_path):
            for contrast_dir in os.listdir(seg_path):
                for seg_group in os.listdir(os.path.join(seg_path, contrast_dir)):
                    group_path = os.path.join(seg_path, contrast_dir, seg_group)
                    if os.path.isdir(group_path):
                        for seg_file in os.listdir(group_path):
                            if seg_file.endswith('.nii') or seg_file.endswith('.nii.gz'):
                                if seg_file.endswith('_old.nii'):
                                    print(f"Skipping {seg_file} as its matrix is reversed in the slice direction.")
                                    continue
                                seg_file_name = seg_file.split('.')[0]
                                seg_target_path = os.path.relpath(os.path.join(group_path, seg_file), root_dir)
                                columns.append(('Segmentations', clean_contrast_name(contrast_dir), seg_group, seg_file_name))
                                data.append((study, seg_target_path))
                    else:
                        print(f"Skipping {seg_group} as it is not a directory in {group_path}")
        else:
            print(f"No Segmentations directory found in {study_path}")

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Study', 'Path'])
    df['Type'], df['Contrast'], df['Info'], df['Part'] = zip(*[(c + (None, None))[:4] for c in columns])

    return df