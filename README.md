# Pix2Gif: Motion-Guided Autoregressive Diffusion for Gif Generation
:orange: \[[Project Page](https://hiteshk03.github.io/)\]

[Hitesh Kandala](https://hiteshk03.github.io/)<sup>1</sup>, [Alexei A. Efros](https://jwyang.github.io/)<sup>2</sup>
<br> Microsoft Research India<sup>1</sup>, Microsoft Research Redmond<sup>2</sup>

## Setup
### Set up python virtual environment
```bash
python3.10 -m venv .pix2gif
source .pix2gif/bin/activate
pip install -r requirements.txt
```

## Demo
This demo takes in an image and a caption and generates a Gif following the input caption. To launch the demo, run:

```bash
python demo.py
```

## Acknowledgement
We build our work on top of [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)

## Citation
<!-- ```
@article{zou2022xdecoder,
  author      = {Zou*, Xueyan and Dou*, Zi-Yi and Yang*, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee*, Yong Jae and Gao*, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
``` -->