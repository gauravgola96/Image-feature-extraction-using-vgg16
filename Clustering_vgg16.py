import os
#os.makedirs("/content/CLUSTER_DATA")

#os.makedirs("/content/CLUSTER_DATA/files")

os.chdir("/content/CLUSTER_DATA/files")
from google.colab import files
files.upload()

from google.colab import files
files.upload()

os.chdir("/content/CLUSTER_DATA")

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.layers import merge, Input

image_input = Input(shape=(224,224,3))

model = VGG16(include_top=False,weights="imagenet",input_tensor=image_input)

model.summary()

data_dir = os.listdir("/content/CLUSTER_DATA/files")



vgg16_feature_list=[]
for i in data_dir:
  img_path ="/content/CLUSTER_DATA/files" +"/"+i
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  vgg16_feature = model.predict(img_data)
  vgg16_feature_np = np.array(vgg16_feature)
  vgg16_feature_list.append(vgg16_feature_np.flatten())


vgg16_feature_list_np = np.array(vgg16_feature_list)

vgg16_feature_list_np.shape

import sklearn
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(vgg16_feature_list_np)

len(kmeans.labels_)

kmeans.cluster_centers_

kmeans.labels_

import pandas as pd

labels = kmeans.labels_

df = pd.DataFrame()

df["files"] = data_dir

df["labels"] = kmeans.labels_

df

cluster_list0=[]
cluster_list1=[]
cluster_list2=[]
cluster_list3=[]

for i in range(len(df)):
  if df.loc[i,"labels"]==0:
    name = df.loc[i,"files"]
    cluster_list0.append(name)
  if df.loc[i,"labels"]==1:
    name = df.loc[i,"files"]
    cluster_list1.append(name)
  if df.loc[i,"labels"]==2:
    name = df.loc[i,"files"]
    cluster_list2.append(name)
  if df.loc[i,"labels"]==3:
    name = df.loc[i,"files"]
    cluster_list3.append(name)
    

os.makedirs("/content/CLUSTER_DATA/cats")
os.makedirs("/content/CLUSTER_DATA/dogs")
os.makedirs("/content/CLUSTER_DATA/horses")
os.makedirs("/content/CLUSTER_DATA/humans")

path = "/content/CLUSTER_DATA/files"
data_dir = os.listdir(path)
for i in data_dir:
    import shutil
    #from shutil 
    if i in cluster_list0:
      source_dir = path +"/" +i
      dest_source = "/content/CLUSTER_DATA/cats"
      shutil.copy(src=source_dir,dst=dest_source)
    if i in cluster_list1:
      source_dir = path +"/" +i
      dest_source = "/content/CLUSTER_DATA/dogs"
      shutil.copy(src=source_dir,dst=dest_source)
    if i in cluster_list2:
      source_dir = path +"/" +i
      dest_source = "/content/CLUSTER_DATA/horses"
      shutil.copy(src=source_dir,dst=dest_source)
    if i in cluster_list3:
      source_dir = path +"/" +i
      dest_source = "/content/CLUSTER_DATA/humans"
      shutil.copy(src=source_dir,dst=dest_source)
      


from google.colab import files

files_cats = os.listdir("/content/CLUSTER_DATA/cats")

# import zipfile
# path_to_zip_file = '/content/data (1).zip'
# directory_to_extract_to=''
# zip_ref = zipfile.ZipFile(path_to_zip_file,"r")
# zip_ref.extractall(directory_to_extract_to)
# zip_ref.close()

import shutil

shutil.make_archive("catsZip", "zip", "/content/CLUSTER_DATA/cats")

import shutil

shutil.make_archive("horsesZip", "zip", "/content/CLUSTER_DATA/horses")
shutil.make_archive("dogsZip", "zip", "/content/CLUSTER_DATA/dogs")
shutil.make_archive("humansZip", "zip", "/content/CLUSTER_DATA/humans")
#shutil.make_archive("catsZip", "zip", "/content/CLUSTER_DATA/cats")

files.download("catsZip.zip")
files.download("dogsZip.zip")
files.download("horsesZip.zip")
files.download("humansZip.zip")
