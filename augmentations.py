###################################
# Distortion
##################
# Photometric Distortion — This includes changing the brightness, contrast, saturation, and noise in an image
# Geometric Distortion — This includes random scaling, cropping, flipping, and rotating.
##################################
# Image Occlusion
##################
# Random Erase — This is a data augmentation technique that replaces regions of the image with random values, or the
# mean pixel value of training set. Typically, it is implemented with varying proportion of image erased and aspect
# ratio of erased area.

# Cutout — square regions are masked during training. Cutout regions are only hidden from the first layer of the CNN.
# This is very similar to random erase, but with a constant value in the overlaid occlusion.

# Hide and Seek — Divide the image into a grid of SxS patches. Hide each patch with some probability (p_hide).

# Grid Mask — Regions of the image are hidden in a grid like fashion.

# MixUp — convex overlaying of image pairs and their labels
# x = Variable(lam * x1 + (1. - lam) * x2)
# y = Variable(lam * y1 + (1. - lam) * y2)

# CutMix — Combine images by cutting parts from one image and pasting them onto the augmented image.

# Mosaic data augmentation — Mosaic data augmentation combines 4 training images into one in certain ratios

# Augmix -- combination of various transforms

import tensorflow as tf
import numpy as np

# after batching images
def cutmix(image, label, PROBABILITY=1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with cutmix applied
  DIM = IMAGE_SIZE[0]
  CLASSES = 104

  imgs = [];
  labs = []
  for j in range(AUG_BATCH):
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    # CHOOSE RANDOM IMAGE TO CUTMIX WITH
    k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
    # CHOOSE RANDOM LOCATION
    x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
    y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
    b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
    WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
    ya = tf.math.maximum(0, y - WIDTH // 2)
    yb = tf.math.minimum(DIM, y + WIDTH // 2)
    xa = tf.math.maximum(0, x - WIDTH // 2)
    xb = tf.math.minimum(DIM, x + WIDTH // 2)
    # MAKE CUTMIX IMAGE
    one = image[j, ya:yb, 0:xa, :]
    two = image[k, ya:yb, xa:xb, :]
    three = image[j, ya:yb, xb:DIM, :]
    middle = tf.concat([one, two, three], axis=1)
    img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
    imgs.append(img)
    # MAKE CUTMIX LABEL
    a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
    if len(label.shape) == 1:
      lab1 = tf.one_hot(label[j], CLASSES)
      lab2 = tf.one_hot(label[k], CLASSES)
    else:
      lab1 = label[j,]
      lab2 = label[k,]
    labs.append((1 - a) * lab1 + a * lab2)

  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
  label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
  return image2, label2

# after batching images
def mosaic(self, index):
  # loads images in a mosaic of 2,3,or 4 combined images

  labels4 = []
  s = self.img_size
  xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
  indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
  for i, index in enumerate(indices):
    # Load image
    img, _, (h, w) = load_image(self, index)

    # place img in img4
    if i == 0:  # top left
      img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
      x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
      x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    elif i == 1:  # top right
      x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
      x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    elif i == 2:  # bottom left
      x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
      x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
    elif i == 3:  # bottom right
      x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
      x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

    img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
    padw = x1a - x1b
    padh = y1a - y1b

    # Labels
    x = self.labels[index]
    labels = x.copy()
    if x.size > 0:  # Normalized xywh to pixel xyxy format
      labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
      labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
      labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
      labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
    labels4.append(labels)

  # Concat/clip labels
  if len(labels4):
    labels4 = np.concatenate(labels4, 0)
    # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
    np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

  # Augment
  # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
  img4, labels4 = random_affine(img4, labels4,
                                degrees=1.98 * 2,
                                translate=0.05 * 2,
                                scale=0.05 * 2,
                                shear=0.641 * 2,
                                border=-s // 2)  # border to remove

  return img4, labels4

# after batching images
def mixup(image, label, PROBABILITY=1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with mixup applied
  DIM = IMAGE_SIZE[0]
  CLASSES = 104

  imgs = [];
  labs = []
  for j in range(AUG_BATCH):
    # DO MIXUP WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
    # CHOOSE RANDOM
    k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
    a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
    # MAKE MIXUP IMAGE
    img1 = image[j,]
    img2 = image[k,]
    imgs.append((1 - a) * img1 + a * img2)
    # MAKE CUTMIX LABEL
    if len(label.shape) == 1:
      lab1 = tf.one_hot(label[j], CLASSES)
      lab2 = tf.one_hot(label[k], CLASSES)
    else:
      lab1 = label[j,]
      lab2 = label[k,]
    labs.append((1 - a) * lab1 + a * lab2)

  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
  label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
  return image2, label2