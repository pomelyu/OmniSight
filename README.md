# Omni-Sight
Collections of vision models

# Face Detection
```bash
# SCRFD
# models can be downloaded from https://github.com/cysin/scrfd_onnx
python -m demo.demo_scrfd \
    --model checkpoints/scrfd_10g_bnkps_shape512x512-237daff4.onnx \
    --image tests/resources/one_girl.jpg
```
