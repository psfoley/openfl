import os
import nibabel as nib
import numpy.ma as ma
import numpy as np
from tqdm import tqdm
import argparse

# \/ Hard Coded arguments
resize = 128  # Final dimension (square), set resize = 240 (original dimension) if no resizing is desired
rotate = 3  # Number of counter-clockwise, 90 degree rotations
# /\ Hard Coded arguments


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

    slices = []
    for slice in range(img.shape[-1]):
        curr = img[:,:,slice]
        slices.append(curr)

    return np.asarray(slices)

    def stack_img_slices(mode_track, stack_order):

        # Put final image channels in the order listed in stack_order

        full_brain = []
        for slice in range(len(mode_track['t1'])):
            current_slice = []
            for mode in stack_order:
                current_slice.append(mode_track[mode][slice,:,:])
            full_brain.append(np.dstack(current_slice))

        # Normalize stacked images (inference will not work if this is not performed)
        stack = np.asarray(full_brain)
        stack = (stack - np.mean(stack))/(np.std(stack))

        return stack


    def resize_data(dataset, new_size):

        # Test/Train images must be the same size

        start_index = int((dataset.shape[1] - new_size)/2)
        end_index = dataset.shape[1] - start_index

        if rotate != 0:
            resized = np.rot90(dataset[:, start_index:end_index, start_index:end_index :], rotate, axes=(1,2))
        else:
            resized = dataset[:, start_index:end_index, start_index:end_index :]

        return resized


    # adapted from https://github.com/NervanaSystems/topologies
def _update_channels(imgs, msks, input_no=1, output_no=1, mode=1, CHANNEL_LAST=True):
    """
    changes the order or which channels are used to allow full testing. Uses both
    Imgs and msks as input since different things may be done to both
    ---
    mode: int between 1-3
    """

    imgs = imgs.astype('float32')
    msks = msks.astype('float32')

    if CHANNEL_LAST:

        shp = imgs.shape

        new_imgs = np.zeros((shp[0],shp[1],shp[2],input_no))
        new_msks = np.zeros((shp[0],shp[1],shp[2],output_no))


        if mode == 1:
            new_imgs[:,:,:,0] = imgs[:,:,:,2] # flair
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
            #print('-'*10,' whole tumor data', '-'*10)
        elif mode == 2:
            #core (non enhancing)
            new_imgs[:,:,:,0] = imgs[:,:,:,0] # t1 post
            new_msks[:,:,:,0] = msks[:,:,:,3]
            #print('-'*10,' enhancing tumor data', '-'*10)
        elif mode == 3:
            #core (non enhancing)
            new_imgs[:,:,:,0] = imgs[:,:,:,1]# t2 post
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,2]+msks[:,:,:,3]# active core
            #print('-'*10,' active core data', '-'*10)

        else:
            new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
    else:
        shp = imgs.shape
        new_imgs = np.zeros((shp[0],input_no, shp[2],shp[3]))
        new_msks = np.zeros((shp[0],output_no, shp[2],shp[3]))

        if mode==1:
            new_imgs[:,0,:,:] = imgs[:,2,:,:] # flair
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,1,:,:]+msks[:,2,:,:]+msks[:,3,:,:]
            #print('-'*10,' Whole tumor', '-'*10)
        elif mode == 2:
            #core (non enhancing)
            new_imgs[:,0,:,:] = imgs[:,0,:,:] # t1 post
            new_msks[:,0,:,:] = msks[:,3,:,:]
            #print('-'*10,' Predicing enhancing tumor', '-'*10)
        elif mode == 3:
            #core (non enhancing)
            new_imgs[:,0,:,:] = imgs[:,1,:,:]# t2 post
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,2,:,:]+msks[:,3,:,:]# active core
            #print('-'*10,' Predicing active Core', '-'*10)

        else:
            new_msks[:,0,:,:] = msks[:,0,:,:]+msks[:,1,:,:]+msks[:,2,:,:]+msks[:,3,:,:]

    return new_imgs, new_msks


def brats17_reader(data_dir, index, label_type):

    # FIXME: put a logging statement next to raised exceptions

    # the name of the directory in root_dir should contain the new institution name (like 'inst12')
    # institution_name = root_dir.split('/')[-1]

    for subdir, dir, files in os.walk(data_dir):
        # Ensure all necessary files are present
        file_root = subdir.split('/')[-1] + "_"
        extension = ".nii.gz"
        img_modes = ["t1","t2","flair","t1ce"]
        need_file = [file_root + mode + extension for mode in img_modes]
        all_there = [(reqd in files) for reqd in need_file]
        if all(all_there):
            mode_track = {mode:[] for mode in img_modes}
            for file in files:
                if file.endswith('seg_binary.nii.gz') or \
                  file.endswith('seg_binarized.nii.gz') or file.endswith('SegBinarized.nii.gz'):
                    path = os.path.join(subdir,file)
                    msk = np.array(nib.load(path).dataobj)
                    parsed = resize_data(parse_segments(msk), resize)
                    msks_all.extend(parsed)

                if file.endswith('t1.nii.gz'):
                    path = os.path.join(subdir,file)
                    img = np.array(nib.load(path).dataobj)
                    mode_track['t1'] = resize_data(parse_images(img), resize)

                if file.endswith('t2.nii.gz'):
                    path = os.path.join(subdir,file)
                    img = np.array(nib.load(path).dataobj)
                    mode_track['t2'] = resize_data(parse_images(img), resize)

                if file.endswith('t1ce.nii.gz'):
                    path = os.path.join(subdir,file)
                    img = np.array(nib.load(path).dataobj)
                    mode_track['t1ce'] = resize_data(parse_images(img), resize)

                if file.endswith('flair.nii.gz'):
                    path = os.path.join(subdir,file)
                    img = np.array(nib.load(path).dataobj)
                    mode_track['flair'] = resize_data(parse_images(img), resize)
        
            brain_imgs = np.asarray(stack_img_slices(mode_track,img_modes))
            this_img = brain_images[idx % 155]
            return this_img

        else:
            # FIXME: here log the presence of incomplete data
            # in this case we will have an incomplete bach, put logic to 
            # handle this in the data_loader?
            raise ValueError("Data in the foler: {} is not complete".format(subdir))                                                                 97,1          29%



