#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import matplotlib as mpl
import matplotlib.pyplot as plt
#from IPython.display import display
#%matplotlib inline

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.transform import rescale
from sklearn.metrics import roc_curve, auc


# In[2]:


import scipy.misc


# In[3]:


labels = pd.read_csv("D:/dataset/b1.csv", index_col=0)
labels.head()
def get_image(row_id, root="D:/dataset"):
     filename = "{}.jpg".format(row_id)
     file_path = os.path.join(root, filename)
     img = Image.open(file_path)
     w=400
     h=400
     im=scipy.misc.imresize(img,(h,w))   
     return np.array(im)
#for i in range(6):
    
bombus_row = labels[labels.genus == 1.0].index[18]

plt.imshow(get_image(bombus_row))
plt.show()

apis_row = labels[labels.genus == 0.0].index[3]
plt.imshow(get_image(apis_row))
plt.show()


# In[4]:


bombus = get_image(bombus_row)

print('Color bombus image has shape: ', bombus)

grey_bombus = rgb2grey(bombus)

plt.imshow(grey_bombus, cmap=mpl.cm.gray)

print('Greyscale bombus image has shape: ', grey_bombus)


# In[5]:


hog_features, hog_image = hog(grey_bombus,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

plt.imshow(hog_image, cmap=mpl.cm.gray)


# In[6]:


def create_features(img):
    color_features = img.flatten()
    grey_image = rgb2grey(img)
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    flat_features = np.hstack(color_features)
    return flat_features

bombus_features = create_features(bombus)

print(bombus_features)


# In[8]:


def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:

        img = get_image(img_id)
        image_features = create_features(img)
        features_list.append(image_features)
        
    feature_matrix = np.array(features_list)
    return feature_matrix

feature_matrix = create_feature_matrix(labels)
print('Feature matrix shape is: ', feature_matrix.shape)


# In[9]:


#Scale feature matrix + PCA
print('Feature matrix shape is: ', feature_matrix.shape)

ss = StandardScaler()
bees_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
bees_pca = ss.fit_transform(bees_stand)
print('PCA matrix shape is: ', bees_pca.shape)


# In[10]:


#Split into train and test sets
X = pd.DataFrame(bees_pca)
y = pd.Series(labels.genus.values)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distrubution of labels in the train set
pd.Series(y_train).value_counts()


# In[11]:


#Train model
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)


# In[12]:


#Score model
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)


# In[13]:


#ROC curve + AUC
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');


# In[ ]:




