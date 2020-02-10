# Vid2Depth

## [Paper](https://sites.google.com/view/vid2depth)

## Usage

### Installation

```bash
pip install tensorflow-gpu==2.1.0
python setup.py install
```

## Train on [BikeDataset](https://sites.google.com/site/brainrobotdata/home/bike-video-dataset)

```bash
wget -O BikeVideoDataset.tar https://storage.googleapis.com/brain-robotics-data/bike/BikeVideoDataset.tar
tar xfv BikeVideoDataset.tar
data_dir=$(pwd)/BikeVideoDataset

vid2depth-train -d $data_dir
# or
python -m vid2depth.bin.train -d $data_dir

# for more information see
vid2depth-train --help
```

## Acknowledgments

Parts of the code where inspired from [Tensorflow models](https://github.com/tensorflow/models/tree/master/research/vid2depth).
