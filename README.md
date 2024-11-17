# Text image dataset
此專案包含兩個資料集，分別為手寫繁體文字影像以及各類Logo圖示透過裁剪與變化字型後得到的文字片段影像:
| ITEM | CHINESE | LOGO |
| :---: | :---: | :---: |
| Number of type | 4803 | 2185 |
| Size | 250809 | 4331 |
| Image | ![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/%E8%B9%8A_78.png) | ![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/2N3501UB___SOT23-page1_Logo0.jpg) |

# Capture text images
透過CRAFT的技術將影像中文字區塊裁切並保留，透過CRAFT所得到的片段區分為熱圖以及文字擷取片段，文字擷取片段即可用於文字辨識之使用。
詳細資訊可參考:https://github.com/clovaai/CRAFT-pytorch

![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/2N3501UB___SOT23-page1_Logo0.jpg)➡️![image](https://github.com/Arthur1114414/Text_image_dataset/blob/main/crop_1.png)
