import os
# import nibabel as nib
import numpy.ma as ma
import numpy as np
import argparse

# FIXME: fix import problem for nibabel

# import sys
# sys.path.append('/home/edwardsb/.local/share/virtualenvs/tfedlearn-cvKTHQG4/lib/python3.5/site-packages')
import nibabel as nib

# for testing performance
import time

def parse_segments(seg, msk_modes):
    # Each channel corresponds to a different region of the tumor, decouple and stack these
    msks_parsed = []
    for slice in range(seg.shape[-1]):
        # which mask values indicicate which label mode
        mode_to_key_value = {"necrotic": 1, "edema": 2, "GD": 4}
        curr = seg[:,:,slice]
        this_msk_parts = []
        for mode in msk_modes:
            this_msk_parts.append(ma.masked_not_equal(curr,mode_to_key_value[mode]).filled(fill_value=0))
        msks_parsed.append(np.dstack(this_msk_parts))

    # Replace all tumorous areas with 1 (previously marked as 1, 2 or 4)
    mask = np.asarray(msks_parsed)
    mask[mask > 0] = 1

    return mask


def normalize_stack(imgs):
    imgs = (imgs - np.mean(imgs))/(np.std(imgs))
    return imgs


def resize_data(dataset, new_size=128, rotate=3):
    """
    Resize 2D images within data by cropping equally from their boundaries.
    
    dataset(np.array): Data containing 2D images whose dimensions are along the 1st 
        and 2nd axes. 
    new_size(int): Dimensions of square image to which resizing will occur. Assumed to be
        an even distance away from both dimensions of the images within dataset. (default 128)
    rotate(int): Number of counter clockwise 90 degree rotations to perform.


    Returns:
        (np.array): resized data
    
    Raises:
        ValueError: If (dataset.shape[1] - new_size) and (dataset.shape[2] - new_size) 
            are not both even integers.

    """

    # Determine whether dataset and new_size are compatible with existing logic
    if (dataset.shape[1] - new_size) % 2 != 0 and (dataset.shape[2] - new_size) % 2 != 0:
        raise ValueError('dataset shape: {} and new_size: {} are not compatible with ' \
            'existing logic'.format(dataset.shape, new_size))

    start_index = int((dataset.shape[1] - new_size)/2)
    end_index = dataset.shape[1] - start_index

    if rotate != 0:
        resized = np.rot90(dataset[:, start_index:end_index, start_index:end_index], 
                           rotate, axes=(1,2))
    else:
        resized = dataset[:, start_index:end_index, start_index:end_index]

    return resized


# adapted from https://github.com/NervanaSystems/topologies
def _update_channels(imgs, msks, img_channels_to_keep, msk_channels_to_keep, channels_last):  
    """
    Filter the channels of images and move placement of channels in shape if desired.
    
    imgs (np.array): A stack of images with channels (channels could be anywhere in number from
    one to four. Images are indexed along first axis, with channels along fourth (last) axis.
    msks (np.array): A stack of binary masks with channels (channels could be anywhere in number from
    one to four. Images are indexed along first axis, with channels along fourth (last) axis.
    img_channels_to_keep (flat np.ndarray): the channels to keep in the image (remove others)
    msk_channels_to_keep (flat np.ndarray): the channels to sum in the mask (resulting in 
    a single channel array) 
    channels_last (bool): Return channels in last axis? otherwise just after first
    """
     
    new_imgs = imgs[:,:,:,img_channels_to_keep]
    # the mask channels that are kept are summed over to leave one channel
    # note the indices producing non-zero entries on these masks are mutually exclusive
    # so that the result continues to be an array with only ones and zeros
    msk_summands = [msks[:,:,:,channel:channel+1] for channel in msk_channels_to_keep]
    new_msks = np.sum(msk_summands, axis=0)
    
    if not channels_last:
        return np.transpose(new_imgs, [0, 3, 1, 2]), np.transpose(new_msks, [0, 3, 1, 2])       
    else:       
        return new_imgs, new_msks


def list_files(root, extension, parts):
    files = [root + part + extension for part in parts]
    return files


def brats17_2d_reader(idx, idx_to_paths, label_type, channels_last=True, 
  numpy_type='float32', normalize_by_task=False):
    """
    Fetch single 2D brain image from disc.

    Assumes data_dir contains only subdirectories, each conataining exactly one 
    of each of the following files: "<subdir>_<type>.nii.gz", where <subdir> is 
    the name of the subdirectory and <type> is one of ["t1", "t2","flair","t1ce"],
    as well as a segmentation label file "<subdir>_<suffix>", where suffix is: 
    'seg_binary.nii.gz', 'seg_binarized.nii.gz', or 'SegBinarized.nii.gz'.
    These files provide all modes and segmenation label for a single patient 
    brain scan consisting of 155 axial slice images. The reader is designed to
    return only one of those images.
    
    Args:
        idx (int or tuple of ints): if int, is the index of a single 2D-image to return
        if tuple of ints, this tuple components need to be 155 appart and so designate
        a full brain to return
        idx_to_paths (list of str): paths to files containing image features and full 
        label set 
        label_type (string): determines way in which label information is combined
        channels_last (bool): Reader should output channels last?, otherwise just after first
        numpy_type (string): The numpy datatype for final casting before return

    Returns:
        np.array: single 2D image associated to the index

    Raises:
        ValueError: If label_type is not in 
                    ['whole_tumor', 'enhanced_tumor', 'active_core', 'other']
        ValueError: If the path determined by idx and indexed_data_paths points 
                    to a file with incomplete data
        ValueError: If idx is neither an int or a tuple of ints, or if it is a tuple
        and the second component is not exactly 155 greater than the first, with the first
        divisible by 155.
    
    """
    # validate that idx is an allowed value, and distill information from idx into
    # the variables idx_start and idx_end. If idx_end is None, this signals the 
    # desire for a single 2D image as return value, otherwise return full brain
    idx_value_error = False
    if not isinstance(idx, int):
        if isinstance(idx, tuple):
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                if (idx[1] - idx[0] == 155) and (idx[0] % 155 == 0):
                    idx_start = idx[0]
                    idx_end = idx[1]
                else:
                    idx_value_error = True
            else: idx_value_error = True
        else: idx_value_error = True
    else:
        idx_start = idx
        idx_end = None

    if idx_value_error:
        raise ValueError("The value of idx is not supported.")

    label_type_to_task = {"whole_tumor": 1, "enhanced_tumor": 2, "active_core": 3, "other": 4}
    try:
        task = label_type_to_task[label_type]
    except KeyError:
        raise ValueError("{} is not a valid label type".format(label_type))
    tasks = label_type_to_task.values()

    subdir = idx_to_paths[idx_start]
    files = os.listdir(subdir)
    # link task number to appropriate image and mask channels of interest
    img_modes = ["t1","t2","flair","t1ce"]
    msk_modes = ["necrotic", "edema", "GD"]
    task_to_img_modes = {1: ["flair"], 2: ["t1"], 3: ["t2"], 4: ["t1","t2","flair","t1ce"]}
    task_to_msk_modes = {
        1: ["necrotic", "edema", "GD"], 
        2: ["GD"], 
        3: ["edema", "GD"], 
        4: ["necrotic", "edema", "GD"]
        }
    msk_names = ["seg_binary", "seg_binarized", "SegBinarized", "seg"]

    # check that all appropriate files are present
    file_root = subdir.split('/')[-1] + "_"
    extension = ".nii.gz"

    # record files needed for each task
    # needed mask files are currntly independent of task, but allowing for dependency here
    need_files_oneof = {task: list_files(file_root, extension, msk_names) for task in tasks}
    if normalize_by_task:
        need_files_all = \
            {task: list_files(file_root, extension, task_to_img_modes[task]) for task in tasks}     
    else:
        need_files_all = \
            {task: list_files(file_root, extension, img_modes) for task in tasks}  

    correct_files = np.all([(reqd in files) for reqd in need_files_all[task]]) and \
      np.sum([(reqd in files) for reqd in need_files_oneof[task]])==1
    if not correct_files:
        raise ValueError("Data in folder: {} incomplete or too many label files.".format(subdir))  

    # get image (features)
    imgs_per_mode = []
    for file in need_files_all[task]:
        path = os.path.join(subdir,file)
        full_brain = np.array(nib.load(path).dataobj)
        imgs_per_mode.append(resize_data(np.transpose(full_brain, [-1, 0, 1])))
    imgs = np.stack(imgs_per_mode, axis=-1)
    imgs = normalize_stack(imgs)  

    # get mask (labels)  
    for file in need_files_oneof[task]:
        if file in files:
            path = os.path.join(subdir,file)
            full_brain_msk = np.array(nib.load(path).dataobj)
            msks = resize_data(parse_segments(full_brain_msk, msk_modes))
            break

    # determine which channels are wanted in our images
    msk_mode_to_channel = {mode: channel_num for (channel_num, mode) in enumerate(msk_modes)}
    task_to_msk_channels = {task: [msk_mode_to_channel[mode] for mode in modes] \
      for (task, modes) in task_to_msk_modes.items()}
    msk_channels_to_keep = np.array(task_to_msk_channels[task])
    # if we normalized by task, we have already restricted the image channels 
    if normalize_by_task:
        img_channels_to_keep = np.arange(imgs.shape[-1])
    else:
        img_mode_to_channel = {mode: channel_num for (channel_num, mode) in enumerate(img_modes)}
        task_to_img_channels = {task: [img_mode_to_channel[mode] for mode in modes] \
          for (task, modes) in task_to_img_modes.items()}
        img_channels_to_keep = np.array(task_to_img_channels[task])
    
    
    if idx_end is None:
        # idx % 155 provides which of the 155 slices to grab from the whole brain
        # here we restrict to a single slice (2D), but preserve dimensions
        slice_idx = idx % 155 
        img = imgs[slice_idx:slice_idx+1, :, :, :]
        msk = msks[slice_idx:slice_idx+1, :, :, :]

    img, msk = _update_channels(img, msk, img_channels_to_keep, msk_channels_to_keep, channels_last)
    
    if idx_end is None:
        # collapsing the one dimensional first axis to produce a 2D image, casting type
        return np.squeeze(img.astype(numpy_type), axis=0), np.squeeze(msk.astype(numpy_type), axis=0)
    else:
        return img.astype(numpy_type), msk.astype(numpy_type)

            



