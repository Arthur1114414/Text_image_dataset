# Text image dataset
This repository contains two data sets, which are handwritten traditional text images and text fragment images of various logo icons obtained by cropping and changing fonts:
| ITEM | CHINESE | LOGO |
| :---: | :---: | :---: |
| Number of type | 4803 | 2185 |
| Size | 250809 | 4331 |
| Image | ![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/%E8%B9%8A_78.png) | ![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/2N3501UB___SOT23-page1_Logo0.jpg) |

# Capture text images
The text blocks in the image are cut and retained through CRAFT technology. The fragments obtained through CRAFT are divided into heat images and text capture fragments. The text capture fragments can be used for text recognition. For detailed information, please refer to:https://github.com/clovaai/CRAFT-pytorch

![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/2N3501UB___SOT23-page1_Logo0.jpg)➡️![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/crop_1.png)

In [CRAFT.py](https://github.com/Arthur1114414/Text_image_dataset/blob/main/CRAFT.py),`image_dir` refers to the directory of the original image, and `output_dir` refers to the image output directory after the text fragment image is captured.
