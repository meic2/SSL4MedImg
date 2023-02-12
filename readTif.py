from PIL import Image
from Image_preprocessing.IP_dermatomyositis_interpolate import read_tiff
from tifffile import TiffFile
from xml.etree import ElementTree
# parser = ElementTree.XMLParser(recover=True)
import numpy as np

path_d = "/scratch/ssc10020/IndependentStudy/dataset/Dermatomyositis/original_data/CD27_Panel_Component/121919_Myo089_[7110,44031]_component_data.tif"
path_l = "/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/170-II-K4.qptiff"
path_l2 = "/scratch/ssc10020/IndependentStudy/dataset/Lupus_Nephritis/170-II-K8.qptiff"

imd = Image.open(path_d)
iml = Image.open(path_l)
iml2 = Image.open(path_l2)

tifd = TiffFile(path_d)
tifl = TiffFile(path_l)
tifl2 = TiffFile(path_l2)

# with TiffFile(path_l) as tif:
#     for page in tif.series[0].pages:
#         print(ElementTree.fromstring(page.description).find('Name').text)

with TiffFile(path_d) as tif:
    for page in tif.series[0].pages:
        print(ElementTree.fromstring(page.description).find('Name').text)

with TiffFile(path_l) as tif:
    for page in tif.series[0].pages:
        print(ElementTree.fromstring(page.description).find('Name').text)

# for page in tifl.series[0].pages:
        # print(ElementTree.fromstring(page.description).find('Name').text)

# for page in tifd.series[0].pages:
        # print(ElementTree.fromstring(page.description).find('Name').text)

import pdb; pdb.set_trace()