# use tensorboard in pytorch

## import a script to visualize

This method only to install cpu-version tensorflow, then input `from logger import Logger` in your python file, `logger = Logger('./logs')` to save tensorboard file.

* **record variable**
```
# (1) Log the scalar values
info = {
    'loss': loss.data[0],
    'accuracy': accuracy.data[0]
}

for tag, value in info.items():
    logger.scalar_summary(tag, value, step)

# (2) Log values and gradients of the parameters (histogram)
for tag, value in model.named_parameters():
    tag = tag.replace('.', '/')
    logger.histo_summary(tag, to_np(value), step)
    logger.histo_summary(tag+'/grad', to_np(value.grad), step)

# (3) Log the images
info = {
    'images': to_np(img.view(-1, 28, 28)[:10])
}

for tag, images in info.items():
    logger.image_summary(tag, images, step)
```

* **watch tensorboard** 
 
input `tensorboard --logdir='./logs' --port=8112` in the current directory, and input `your_ip:8112` in your browser to get into tensorboard.
