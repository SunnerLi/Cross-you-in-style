# Crossing You in Style: Cross-modal Style Transfer from Music to Visual Arts

![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg) 
![Pytorch 0.4.1](https://img.shields.io/badge/pytorch-0.4.1-red.svg)
![torchvision 0.2.1](https://img.shields.io/badge/torchvision-0.2.1-red.svg)
![librosa 0.7.1](https://img.shields.io/badge/librosa-0.7.1-green.svg)


[[Project page]](https://sunnerli.github.io/Cross-you-in-style/) [[Arxiv paper]](https://arxiv.org/abs/2009.08083)
[[Dataset]](https://drive.google.com/drive/folders/1XgrXx1qKd8etj9-75ma_8z1tlO8Y49tE)

![](https://raw.githubusercontent.com/SunnerLi/Cross-you-in-style/gh-pages/resources/teaser.png)

The official PyTorch implementation of our [ACM Multimedia 2020](https://2020.acmmm.org/) paper. With our proposed framework, we can stylized the given image with another condition music piece.

Abstract
---
> Music-to-visual style transfer is a challenging yet important cross-modal learning problem in the practice of creativity. Its major difference from the traditional image style transfer problem is that the style information is provided by music rather than images. Assuming that musical features can be properly mapped to visual contents through semantic links between the two domains, we solve the music-to-visual style transfer problem in two steps: music visualization and style transfer. The music visualization network utilizes an encoder-generator architecture with a conditional generative adversarial network to generate image-based music representations from music data. This network is integrated with an image style transfer method to accomplish the style transfer process. Experiments are conducted on WikiArt-IMSLP, a newly compiled dataset including Western music recordings and paintings listed by decades. By utilizing such a label to learn the semantic connection between paintings and music, we demonstrate that the proposed framework can generate diverse image style representations from a music piece, and these representations can unveil certain art forms of the same era. Subjective testing results also emphasize the role of the era label in improving the perceptual quality on the compatibility between music and visual content.

Paper
---
Please cite our paper if you think our research or dataset for your research. \* indicates equal contributions

[Cheng-Che Lee*](https://sunnerli.github.io/), [Wan-Yi Lin*](https://github.com/boop477), [Yen-Ting Shih](#), [Pei-Yi Patricia Kuo](https://www.iss.nthu.edu.tw/faculty/Pei-Yi-Patricia-Kuo), and [Li Su](https://www.iis.sinica.edu.tw/pages/lisu/index_en.html), "Crossing You in Style: Cross-modal Style Transfer from Music to Visual Arts", in ACM International Conference on Multimedia, 2020.


```bibtex
@inproceedings{lee2020crossing,
  title={Crossing You in Style: Cross-modal Style Transfer from Music to Visual Arts},
  author={Lee, Cheng-Che and Lin, Wan-Yi and Shih, Yen-Ting and Kuo, Pei-Yi and Su, Li},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3219--3227},
  year={2020}
}
```

Method
---
![](https://raw.githubusercontent.com/SunnerLi/Cross-you-in-style/gh-pages/resources/overview.png)

Prerequisite
---
* torch 0.4.1
* torchvision 0.2.1
* librosa 0.7.1
* python 3.5.2
* cupy (for linear style transfer)
* pynvrtc (for linear style transfer)

## Model Evaluation
### Generate Music Style Representation
1. Download the pretrained [model](https://drive.google.com/file/d/1drEpCIA0UapXAqRyk1_8_gm4LV6A2U5x/view?usp=sharing); place the model in `./Source`.
2. Place the target `.wav` file to `./Source`;
3. Generate `./Source/clips.json`, which contains:
    ```
    [
        {
            "third": <Start reading at this time>
            "name": <The name of the audio>
            "seg_idx": <The unique index of this segment. The music style representation of this segment will be <seg_idx>.jpg>
            "path": <The path to the audio>
        },
        {...},
        ...c
    ]
    ```
4. `bash evaluate.sh <base> <count>`
* Parameters:
    * `base`: Integer. Music style representations will be inferenced for `<count>` times, and results will be saved to `Results/<wav name>/Style_sample<base+count>`
    * `count`: Integer. Music representations will be inferenced for `<count>` times, and results will be saved to `Results/<wav name>/Style_sample<base+count>`
* Output:
    * `Results/<wav name>/Style_sample<base+count>`
    
```
Example
> Folder structure:
    Source/Spring.wav
    Source/last2.pth
    Source/clips.json
        // The content of Source/clips.json
        [
            {
                "third": 2.14,
                "name": "Spring",
                "seg_idx": 1,
                "path": ./Source/spring.wav
            },
            {
                "third": 5.72,
                "name": "Spring",
                "seg_idx": 2,
                "path": ./Source/spring.wav
            },
            ...
        ]
> bash evaluate.sh 0 2
> Output 
    Results/Spring/Style_sample00
    Results/Spring/Style_sample01
```
    
### Style Transfer
#### Super Resolution
1. We use `ESRGAN` to raise the resolution of the music style representation. Clone the [repository](https://github.com/xinntao/ESRGAN) and follow the instruction to download the pretrained model.
2. Download the modified [`test.py`](https://drive.google.com/file/d/1GLK_KVR9TQ2uZ-YIN8d_ewOQSOG9tneT/view?usp=sharing) and replace the original one. 

#### Linear Style Transfer
1. Clone the [repository](https://github.com/sunshineatnoon/LinearStyleTransfer) and follow the instruction to download the pretrained model and compile the pytorch_spn repository.
2. Download the modified [`TestPhotoReal.py`](https://drive.google.com/file/d/1q4QjNOjxZltx-5q845hkhzOREvXt5LFG/view?usp=sharing) and replace the original one.
3. Download the modified [`LoaderPhotoReal.py`](https://drive.google.com/file/d/1JvqM0aj_Tcibjq5-z-cOkgzD8UF2m_mC/view?usp=sharing) and replace the original one located in `libs` 

#### Evaluate
`python batch_paint.py --content_image <path1> --style_images <path2>`

* Parameters:
    * `--content_image`: The path of the content image.
    * `--style_images`: The path to the folder where the music style representations stay.
* Output:
    * `<image name>/Content` : The content image.
    * `<image name>/LR` : Music representations in low resolution.
    * `<image name>/HR` : Music representations in high resolution.
    * `<image name>/Result` : The result of phto-realistic style transfer.
    * `<image name>/filtered` : Copies of `<image name>/Result/*_filtered.jpg`.
    * `<image name>/smooth` : Copies of `<image name>/Result/*_smooth.jpg`.
    * `<image name>/transfer` : Copies of `<image name>/Result/*_transfer.jpg`.

```
Example
> Folder structure:
    ./Source/
    ./ESRGAN/
    ./LinearStyleTransfer/
    ./Results/
    ./content.jpg
> python batch_paint.py --content_image content.jpg --style_images Results/Spring/Style_sample00
> Output 
    content/Content/
    content/LR/
    content/HR/
    content/LR/
    content/Result/
    content/filtered/
    content/smooth/
    content/transfer/
```
