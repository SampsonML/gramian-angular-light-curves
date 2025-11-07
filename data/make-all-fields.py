import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import pandas as pd
import tqdm
import glob
import cv2


def create_transform(data, size, method='G'):
    if method == 'G':
        transformer = GramianAngularField(method='summation')
    elif method == 'D':
        transformer = GramianAngularField(method='difference')
    elif method == 'M':
        transformer = MarkovTransitionField()
    elif method == 'R':
        transformer = RecurrencePlot()
    else:
        raise ValueError("Invalid method. Choose from 'G', 'D', 'M', or 'R'.")
    
    transformed_data = transformer.fit_transform(np.expand_dims(data, axis=0))[0]

    # Resize to desired size if necessary 
    if transformed_data.shape[0] < size:
        pad_width = size - transformed_data.shape[0]
        transformed_data = np.pad(transformed_data, ((0, pad_width), (0, pad_width)), mode='constant', constant_values=0)

    # Resize down if larger than size
    elif transformed_data.shape[0] > size:
        transformed_data = cv2.resize(transformed_data, (size, size), interpolation=cv2.INTER_AREA)

    return transformed_data


# load in metadata 
meta_data = pd.read_csv("plasticc_train_metadata.csv")

# creat label for object class
label_dict = {90: 'SNIa', 
              67: 'SNIa-91bg', 
              52: 'SNIax',
              42: 'SNII', 
              62: 'SNIbc',
              95: 'SLSN-I',
              15: 'TDE',
              64: 'KN',
              88: 'AGN',
              92: 'RRL',
              65: 'M-dwarf',
              16: 'EB',
              53: 'Mira',
              6: 'mu Lens-Single',
              991: 'mu Lens-Binary',
              992: 'ILOT',
              993: 'CaRT',
              994: 'PISN',
              995: 'mu Lens-String'}

data_dict = []
# now make a light curve plot for every object in training curves folder
#for filename in tqdm.tqdm(glob.glob('training_no_transform/*')):
for filename in tqdm.tqdm(glob.glob('test_no_transform/*')):
    # load the numpy file
    data = pd.read_csv(filename)

    # get the object name
    obj_name = filename.split('/')[-1].split('.')[0]
    ID = int(obj_name)
    obj_class = meta_data[meta_data['object_id'] == ID]['target'].values[0]
    label_id = obj_class
    obj_class = label_dict[obj_class]
    # get the object data 
    lab1 = lab2 = lab3 = lab4 = lab5 = lab6 = False # for legend
    # the data 
    u_band = np.array(data['u_flux']) 
    u_time = np.array(data['u_time']) 
    g_band = np.array(data['g_flux'])
    g_time = np.array(data['g_time'])
    r_band = np.array(data['r_flux'])
    r_time = np.array(data['r_time'])
    i_band = np.array(data['i_flux'])
    i_time = np.array(data['i_time'])
    z_band = np.array(data['z_flux'])
    z_time = np.array(data['z_time'])
    y_band = np.array(data['y_flux'])
    y_time = np.array(data['y_time'])

    # remove all nans 
    u_band = u_band[~np.isnan(u_band)]
    g_band = g_band[~np.isnan(g_band)]
    r_band = r_band[~np.isnan(r_band)]
    i_band = i_band[~np.isnan(i_band)]
    z_band = z_band[~np.isnan(z_band)]
    y_band = y_band[~np.isnan(y_band)]

    # make the gramian angular field for each band 
    size = 40
    u_gaf = create_transform(u_band, size=size, method='G')
    g_gaf = create_transform(g_band, size=size, method='G')
    r_gaf = create_transform(r_band, size=size, method='G')
    i_gaf = create_transform(i_band, size=size, method='G')
    z_gaf = create_transform(z_band, size=size, method='G')
    y_gaf = create_transform(y_band, size=size, method='G')

    # make markov transition field for each band 
    u_mtf = create_transform(u_band, size=size, method='M')
    g_mtf = create_transform(g_band, size=size, method='M')
    r_mtf = create_transform(r_band, size=size, method='M')
    i_mtf = create_transform(i_band, size=size, method='M')
    z_mtf = create_transform(z_band, size=size, method='M')
    y_mtf = create_transform(y_band, size=size, method='M')

    # stack into 12 channel image 
    multi_channel_image = np.stack([u_gaf, g_gaf, r_gaf, i_gaf, z_gaf, y_gaf, 
                                    u_mtf, g_mtf, r_mtf, i_mtf, z_mtf, y_mtf], axis=-1)

    # now create an image label pair and save 
    data_dict.append({'image': multi_channel_image, 'label': label_id, 'class': obj_class}) 

    #if len(data_dict) >= 50:
    #    break

# Extract from your data_dict
images = np.array([d['image'] for d in data_dict])
labels = np.array([d['label'] for d in data_dict])
classes = np.array([d['class'] for d in data_dict])

np.savez('training-data/gaf-mtf-test_images_labels.npz', X=images, y=labels, class_names=classes)

# test the load in 
data = np.load('training-data/gaf-mtf-test_images_labels.npz')
X = data['X']
y = data['y']
class_names = data['class_names']

print(f"Loaded {X.shape[0]} images with shape {X.shape[1:]}")

