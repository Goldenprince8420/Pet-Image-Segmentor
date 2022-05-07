# @title ****Result Function****

def show_results(integer_slider, test_df = test_df, info = info, BATCH_SIZE = BATCH_SIZE):
  y_true_images, y_true_segments = get_test_image_and_annotation_arrays()

  results = model.predict(test_df, steps=info.splits['test'].num_examples//BATCH_SIZE)
  results = np.argmax(results, axis=3)
  results = results[..., tf.newaxis]

  cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_true_segments, results)

  print("__________________________________Class Wise Metrics________________________")

  for idx, iou in enumerate(cls_wise_iou):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, iou)) 

  for idx, dice_score in enumerate(cls_wise_dice_score):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, dice_score)) 

  y_pred_mask = make_predictions(y_true_images[integer_slider], y_true_segments[integer_slider])

  iou, dice_score = class_wise_metrics(y_true_segments[integer_slider], y_pred_mask)

  print("__________________________Results for Given Image__________________")  

  display_with_metrics([y_true_images[integer_slider], y_pred_mask, y_true_segments[integer_slider]], iou, dice_score)
