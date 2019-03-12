from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images,
extract_labels

with open('~/media_samples/t10k-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('~/media_samples/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f)
