from unet_model import *
from data import *

# @title ****Prediction Functions****


def get_test_image_and_annotation_arrays(test_dataset, info, BATCH_SIZE):
    ds = test_dataset.unbatch()
    ds = ds.batch(info.splits['test'].num_examples)
    images = []
    y_true_segments = []
    for image, annotation in ds.take(1):
        y_true_segments = annotation.numpy()
        images = image.numpy()
    y_true_segments = y_true_segments[:(info.splits['test'].num_examples -
                                        (info.splits['test'].num_examples % BATCH_SIZE))]
    return images[:(info.splits['test'].num_examples -
                    (info.splits['test'].num_examples % BATCH_SIZE))], y_true_segments


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()


def make_predictions(model, image, mask, num=1):
    image = np.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)

    return pred_mask
