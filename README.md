# Noise2Noise
Tensorflow [Noise2Noise](https://arxiv.org/abs/1803.04189) implementation.

Noise2Noise is a machine learning algorithm that can learn signal reconstruction from only
noisy examples, i.e. both inputs and targets are noisy realisations of the same image.

## Setup
* Install nvidia-docker
* Create tfrecords
```bash
$ docker build -t n2n .
```

## Training
```bash
$ python -m n2n.train --helpfull
```

```bash
$ python -m n2n.train \
    --train_files "train files pattern" \
    --eval_files "eval files pattern" \
    --model_dir path/to/model_dir
```

Or via docker
```bash
$ ./scripts/run-in-docker <command>
```

## Results
TODO: Show loss curves and images.

```bash
$ python -m n2n.train <required-args> --noise additive_gaussian --loss l2
```
TODO

```bash
$ python -m n2n.train <required-args> --noise text --loss l1
```
TODO



