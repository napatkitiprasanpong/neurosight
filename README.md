# Models

NeuroSight's pose mode uses **MoveNet SinglePose Lightning** (Google, Apache-2.0 licence)
in ONNX format. The file is not included. Put it here as:

```
models/movenet_lightning.onnx
```

## Get the model (one time, needs Python)

TensorFlow Hub models have been moving to Kaggle Models. If a link below fails, search
for "MoveNet SinglePose Lightning" on https://www.kaggle.com/models/google/movenet and
download the TensorFlow 2 (saved model) version.

```
pip install tensorflow tf2onnx
curl -L -o movenet.tar.gz "https://tfhub.dev/google/movenet/singlepose/lightning/4?tf-hub-format=compressed"
mkdir movenet && tar -xzf movenet.tar.gz -C movenet
python -m tf2onnx.convert --saved-model movenet --output movenet_lightning.onnx
```

Then move `movenet_lightning.onnx` into this folder. If the conversion complains about the
opset, add `--opset 13`.

## What the program expects

- Input: `[1, 192, 192, 3]` RGB, values 0-255 (int32; float32 is also tried automatically)
- Output: `[1, 1, 17, 3]` = 17 keypoints, each `[y, x, confidence]` between 0 and 1

To use MoveNet Thunder (256x256, slower, more accurate) the input size constant in
`crates/perception` and the app must be changed to 256.
