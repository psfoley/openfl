import os
# import nibabel as nib
import numpy.ma as ma
import numpy as np
import argparse

# FIXME: Put docstrings in functions.

import sys
sys.path.append('/home/edwardsb/.local/share/virtualenvs/tfedlearn-cvKTHQG4/lib/python3.5/site-packages')
import nibabel

def parse_segments(seg):

    # Each channel corresponds to a different region of the tumor, decouple and stack these

    msks_parsed = []
    for slice in range(seg.shape[-1]):
        curr = seg[:,:,slice]
        GD = ma.masked_not_equal(curr,4).filled(fill_value=0)
        edema = ma.masked_not_equal(curr,2).filled(fill_value=0)
        necrotic = ma.masked_not_equal(curr,1).filled(fill_value=0)
        none = ma.masked_not_equal(curr,0).filled(fill_value=0)

        msks_parsed.append(np.dstack((none,necrotic,edema,GD)))

    # Replace all tumorous areas with 1 (previously marked as 1, 2 or 4)
    mask = np.asarray(msks_parsed)
    mask[mask > 0] = 1

    return mask




def parse_images(img):
    # FIXME: Why not just use np.transpose here? Testing below (with assert) for equivalence.

    slices = []
    for slice in range(img.shape[-1]):
        curr = img[:,:,slice]
        slices.append(curr)

    assert np.asarray(slices) == np.transpose(img, [-1, 0, 1])

    return np.asarray(slices)


def stack_img_slices(mode_track, stack_order):
    # FIXME: Why not just use concatenate? Testing below (with assert) for equivalence
    # Put final image channels in the order listed in stack_order

    full_brain = []
    for slice in range(len(mode_track['t1'])):
        current_slice = []
        for mode in stack_order:
            current_slice.append(mode_track[mode][slice,:,:])
        full_brain.append(np.dstack(current_slice))
    image_stack = np.asarray(full_brain)

    full_brain_too = []
    for mode in stack_order:
        full_brain_too.append(np.expand_dims(mode_track[mode], axis=-1))
    image_stack_too = np.concatenate(full_brain_too, axis=-1)

    assert image_stack == image_stack_too

    return image_stack

def normalize_stack(image_stack):
    """

    """

    # Normalize stacked images (inference will not work if this is not performed)
    image_stack = (image_stack - np.mean(image_stack))/(np.std(image_stack))

    return image_stack



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
        resized = np.rot90(dataset[:, start_index:end_index, start_index:end_index, :], 
                           rotate, axes=(1,2))
    else:
        resized = dataset[:, start_index:end_index, start_index:end_index, :]

    return resized


# adapted from https://github.com/NervanaSystems/topologies
def _update_channels(imgs, msks, task=1, channel_last=True):  
    """
    Combine and reorder channels with independependent logic for imgs and msks.
    
    imgs (np.array): 2D four channel feature images indexed along first axis, with 
        channels along fourth axis.
    msks (np.array): 2D four channel label images (masks) indexed along first axis, with 
        channels along fourth axis.
    task (int): Determines method of channel combination (can be 1-4) (default 1)
    channel_last (bool): Determines whether to keep channels last or move to just 
        after first. (default True)
    """
    
    imgs = imgs.astype('float32')
    msks = msks.astype('float32')

    if channel_last:
        shp = imgs.shape
        new_imgs = np.zeros((shp[0],shp[1],shp[2], 1))
        new_msks = np.zeros((shp[0],shp[1],shp[2], 1))


        if task == 1:
            new_imgs[:,:,:,0] = imgs[:,:,:,2] # flair
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
        elif task == 2:
            new_imgs[:,:,:,0] = imgs[:,:,:,0] # t1 post
            new_msks[:,:,:,0] = msks[:,:,:,3]
        elif task == 3:
            new_imgs[:,:,:,0] = imgs[:,:,:,1]# t2 post
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,2]+msks[:,:,:,3]# active core
        elif task == 4:
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]

        else:
            raise ValueError("{} is not a valid task value".format(task))
    else:
        shp = imgs.shape
        new_imgs = np.zeros((shp[0], 1, shp[2],shp[3]))
        new_msks = np.zeros((shp[0], 1, shp[2],shp[3]))

        if task == 1:
            new_imgs[:,0,:,:] = imgs[:,2,:,:] # flair
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,1,:,:]+msks[:,2,:,:]+msks[:,3,:,:]
        elif task == 2:
            new_imgs[:,0,:,:] = imgs[:,0,:,:] # t1 post
            new_msks[:,0,:,:] = msks[:,3,:,:]
        elif task == 3:
            new_imgs[:,0,:,:] = imgs[:,1,:,:]# t2 post
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,2,:,:]+msks[:,3,:,:]# active core
            
        elif task == 4:
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,1,:,:]+msks[:,2,:,:]+msks[:,3,:,:]

        else:
            raise ValueError("{} is not a valid mode value".format(task))

    return new_imgs, new_msks


def brats17_2d_reader(idx, indexed_data_paths, label_type):
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
        idx (int): index of image
        indexed_data_paths (list of str): paths to files containing image features and full 
        label set label_type (string): determines way in which label information is combined

    Returns:
        np.array: single 2D image associated to the index

    Raises:
        ValueError: If label_type is not in 
                    ['whole_tumor', 'enhanced_tumor', 'active_core', 'other']
        ValueError: If the path determined by idx and indexed_data_paths points 
                    to a file with incomplete data
    
    """

    # FIXME: put a logging statement next to raised exceptions

    if label_type == 'whole_tumor':
        task = 1
    elif label_type == 'enhanced_tumor':
        task = 2
    elif label_type == 'active_core':
        task = 3
    elif label_type == 'other':
        task = 4
    else:
        raise ValueError("{} is not a valid label type".format(label_type))


    subdir = indexed_data_paths[idx]
    file_root = subdir.split('/')[-1] + "_"
    for files in os.list_dir(subdir):
        # Ensure all necessary files are present
        # FIXME: Allow for fewer files depending on mode requested? Change all_there to account
        # for mode, only grab needed files and modify logic in _update_channels moving 
        # appropriate logic down here.        
        extension = ".nii.gz"
        img_modes = ["t1","t2","flair","t1ce"]
        need_file = [file_root + mode + extension for mode in img_modes]
        all_there = [(reqd in files) for reqd in need_file]
        if all(all_there):
            track_mode = {mode:[] for mode in img_modes}
            for file in files:
                if file.endswith('seg_binary.nii.gz') or \
                file.endswith('seg_binarized.nii.gz') or file.endswith('SegBinarized.nii.gz'):
                    path = os.path.join(subdir,file)
                    full_brain_msk = np.array(nib.load(path).dataobj)
                    full_brain_msk = resize_data(parse_segments(msk))

                if file.endswith('t1.nii.gz'):
                    path = os.path.join(subdir,file)
                    full_brain_img_t1 = np.array(nib.load(path).dataobj)
                    track_mode['t1'] = resize_data(parse_images(full_brain_img_t1))

                if file.endswith('t2.nii.gz'):
                    path = os.path.join(subdir,file)
                    full_brain_img_t2 = np.array(nib.load(path).dataobj)
                    track_mode['t2'] = resize_data(parse_images(full_brain_img_t2))

                if file.endswith('t1ce.nii.gz'):
                    path = os.path.join(subdir,file)
                    full_brain_img_t1ce = np.array(nib.load(path).dataobj)
                    track_mode['t1ce'] = resize_data(parse_images(full_brain_img_t1ce))

                if file.endswith('flair.nii.gz'):
                    path = os.path.join(subdir,file)
                    full_brain_img_flair = np.array(nib.load(path).dataobj)
                    track_mode['flair'] = resize_data(parse_images(full_brain_img_flair))

            # FIXME: We had a np.asarray() surrounding below, do we need this really? asserting to see
            full_brain_img = normalize_stack(stack_img_slices(track_mode,img_modes))
            assert np.asarray(full_brain_img) == full_brain_img
            
            # floor(idx/155) is the patient, idx % 155 provides which of the 155 slices to grab
            # Objecgts img and msk are each a 2D image, but have an additional axis to allow for 
            # _update_channels to process 3D images (the expected use case for some models).
            img, msk = _update_channels(np.expand_dims(full_brain_img[idx % 155], axis=0), 
                                        np.expand_dims(full_brain_msk[idx % 155], axis=0), 
                                        task=task)

            # collapsing the length one first axis
            return img[0], msk[0]

        else:
            # FIXME: here log the presence of incomplete data
            # Collect rest of batch anyway, and put logic to handle this in the data_loader?
            raise ValueError("Data in the foler: {} is not complete.".format(subdir))



