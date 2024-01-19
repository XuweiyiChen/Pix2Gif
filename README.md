# Pix2Gif: Motion-Guided Autoregressive Diffusion for Gif Generation
![teaser_op](https://github.com/hiteshK03/Pix2Gif/assets/45922320/3d8ca72e-b2f3-48fd-a732-574987454fd3)

<!-- :grapes: \[[arXiv](https://github.com/hiteshK03/Pix2Gif/)\] &nbsp;  -->
:orange: \[[Project Page](https://hiteshk03.github.io/Pix2Gif/)\]

[Hitesh Kandala](https://hiteshk03.github.io/)<sup>1</sup>, [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)<sup>2</sup>, [Jianwei Yang](https://jwyang.github.io/)<sup>2</sup>
<br> Microsoft Research India<sup>1</sup>, Microsoft Research Redmond<sup>2</sup>

## Setup
### Set up python virtual environment
```bash
python3.10 -m venv .pix2gif
source .pix2gif/bin/activate
pip install -r requirements.txt
```

## Demo
![github_demo](https://github.com/hiteshK03/Pix2Gif/assets/45922320/e3b1605c-b8e2-4ab7-8329-17d0b611e68b)
This demo takes in an image and a caption and generates a Gif following the input caption. To launch the demo, run:
```bash
python demo.py
```

## Acknowledgement
We build our work on top of [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)

## Citation
```bibtex
@article{kandala2024pix2gif,
    title={Pix2Gif: Motion-Guided Autoregressive Diffusion for Gif Generation},
    author={Kandala, Hitesh and Gao, Jianfeng and Yang, Jianwei},
    journal={arXiv preprint},
    year={2024}
  }
