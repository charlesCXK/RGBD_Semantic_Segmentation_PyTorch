# SA-Gate

Official PyTorch implementation of "Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation" ([ECCV, 2020](http://eccv2020.eu/)).

<img src='pic/arch.pdf'>

## Set Up

Coming soon.

## File Structure

```
./
-- furnace
-- DATA
---- pytorch-weight
---- NYUDepthv2
-------- ColoredLabel
-------- Depth
-------- HHA
-------- Label
-------- RGB
-------- test.txt
-------- train.txt
```

## Data preparation

#### NYU Depth V2

You could download the official NYU Depth V2 data [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). After downloading the official data, you should modify them according to the structure of directories we provide. We also provide the processed data (link will be updated soon). *We will delete the link at any time if the owner of NYU Depth V2 requests*.

#### How to generate HHA maps?

If you want to generate HHA maps from Depth map, please refer to [https://github.com/charlesCXK/Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python).

## Citation

Please consider citing this project in your publications if it helps your research.

```
@inproceedings{chen2020SAGate,
  title={Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation},
  author={Chen, Xiaokang and Lin, Kwan-Yee and Wang, Jingbo and Wu, Wayne and Qian, Chen and Li, Hongsheng and Zeng, Gang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Acknowledgement

Thanks [TorchSeg](https://github.com/ycszen/TorchSeg) for their excellent project!

## TODO

- [ ] More encoders such as HRNet.
- [ ] Code and data for Cityscapes.