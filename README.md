# OxfordFlowersClassifier
An image classifier in TensorFlow built on top of a pre-trained network [MobileNet](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4).
By transfer learning, MobileNet was trained to classify [the Oxford Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
Both MobileNet and the Oxford Flowers dataset are available through TensorFlow Hub and TensorFlow Datasets, respectively.

The new network was trained fairly quickly in just 20 epochs, reaching a validation accuracy above 70%.

The best model in terms of validation accuracy is provided in HDF5 format "best_model.h5"

The file "label_map.json" maps the integer labels to the flower names.

The folder "test_images" comtains 4 flower images used for sanity checks throughout the notebook.

"predict.py" can be run from the shell to perform inference on a given image using the best model, unless a different model is specified by the user in the command line.
