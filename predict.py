def main():

    from PIL import Image
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers
    import tensorflow_hub as hub
    import argparse
    import json
    import warnings
    warnings.filterwarnings('ignore')


    def process_image(img_arr):
        img_out = tf.convert_to_tensor(img_arr)
        size = [224,224]
        img_out = tf.image.resize(img_out,size)
        img_out = tf.cast(img_out, tf.float32)
        img_out /= 255
        img_out = img_out.numpy()
        return img_out

    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help = "Path of the test image")
    parser.add_argument("model_file", help = "File where the model is stored")
    parser.add_argument("--top_k", help = "Return the k topmost likely classes", type=int, default = 5)
    parser.add_argument("--category_names", help = "Path to JSON file mapping labels to category names", default = "label_map.json")

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_file, custom_objects={'KerasLayer':hub.KerasLayer})

    im = np.asarray(Image.open(args.image_path))
    im = process_image(im)
    im = np.expand_dims(im, axis=0)
    probs = model.predict(im).squeeze()
    top_idx = np.argsort(probs).squeeze()[-args.top_k:][::-1].tolist()
    top_values = [probs[i] for i in top_idx]

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    top_classes=[]

    for i in range(args.top_k):
        top_classes.append(class_names[str(top_idx[i]+1)])

    print(top_values)
    print(top_classes)

if __name__ == "__main__":
    main()
