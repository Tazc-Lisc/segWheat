Corn Image Segmentation
This is a toy program that can segment images of corn and estimate what percentage of the kernels have been consumed.

Installation
To use this program, you must first install the requirements. It is recommended that you do this in a virtual environment. Note that this code was tested with Python 3.7.

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
Testing on Images
Segmenting an image can be done as follows:

python train.py examples
This will bring up a window showing the segmentation masks for each ear of corn found in the image. It will also print out the corresponding fraction of the kernels that have been consumed.

Testing Sharpmask
Using the Sharpmask-based segmenter is very similar to using the classical one. Note that, before you do this, you will need to download the Sharpmask weights from here. Unzip the resulting file.

python train.py examples
You will need to replace the last argument with the path to the weights that you downloaded.# segWheat
code for Wheat segment
