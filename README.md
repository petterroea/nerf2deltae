# Nerf 2 Delta-E

I wanted some more sturdy image validation tools for validating my NeRF models, other than PSNR and Loss. This tool automatically calculates Delta-E for all test outputs in a NeRF output directory. This requires that the `--render_test` is used, as you need the output to have the same poses as the ground truth images.

Made to work with [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

## Features

 * Parses blender dataset files used by NeRF(or at least nerf-pytorch)
 * Automatically calculates Delta-e for the test output from NeRF

## ⚠️Caveats⚠️

 * Only the blender dataset type is supported, but expanding functionality to other datasets would be very easy to do! *wink wink*
 * Only `-render_test` outputs are supported

## Getting started

Install packages:

```
pip install -r requirements.txt
```

Run the program:

```
python nerf2deltae.py ~/my-nerf-datasets/my-dataset ~/nerf-pytorch/logs/my_nerf_model/testset_050000/
```

Observe the output:

```
Loading from /home/petterroea/mitsuba_renders/nerf_teapot_intense_goniochromatic
Loading masks
Loading from /home/petterroea/mitsuba_renders/nerf_mask/
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/003.png: 19.667814
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/005.png: 14.068017
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/001.png: 21.886508
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/004.png: 14.316636
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/002.png: 14.063049
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/000.png: 17.839711
Delta-E for /tmp/teapot_intense_goniochromatic/testset_050000/006.png: 6.081055
###################
# Delta-e report: #
###################
Min Delta-E: 6.081055
Max Delta-E: 21.886508
Average Delta-E: 15.417542
Standard deviation: 4.748189
```

## Usage

```
usage: nerf2deltae.py [-h] [--testskip TESTSKIP] [--dataset_type {blender}] [--mask_dir MASK_DIR] dataset testset

Reads a vanilla NeRF dataset, calculates what images are used in the test dataset, and then calculates delta e over those images, given
the ground truth. Only works on scenes rendered with render_test. Can optionally use mask_dir to only assess pixels that you want to use

positional arguments:
  dataset               Dataset directory for the model
  testset               Output from nerf

options:
  -h, --help            show this help message and exit
  --testskip TESTSKIP   Testskip factor, defaults to 8
  --dataset_type {blender}
                        Dataset type
  --mask_dir MASK_DIR   Object mask directory

```