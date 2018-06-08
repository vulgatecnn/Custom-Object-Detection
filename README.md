是一个比较好的入门的例子
熟悉搭建环境和如何训练

This is a good case for learn TF 


几点说明
1.faster config的文件内容见下 ( I modify faster config ,example see below)
2.我在ubuntu 下运行ok，但是windows下提示step 不能是0. Under ubuntu 18 , it's worked, but under windows 10 ,the script display step 0 error 
3.fix  export model script
python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix train/model.ckpt-20572 \
        --output_directory output_inference_graph
        
Attaction:
     3.1 After trained_checkpoint_prefix , you should add the floder name, "TRAIN"
     3.2 In train folder, you will see a lot of ckpt file , I choice the second largest ckpt file. If I choice the largest ckp file ,the script report a error.
    
     
     

-rwxrwxrwx 1 vulgate vulgate       277 6月   8 11:55 checkpoint*
-rwxrwxrwx 1 vulgate vulgate  15306196 6月   7 09:30 events.out.tfevents.1528334980.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  14753252 6月   7 09:33 events.out.tfevents.1528335171.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  14753248 6月   7 09:34 events.out.tfevents.1528335230.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  19438109 6月   7 09:46 events.out.tfevents.1528335490.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  17017802 6月   7 10:40 events.out.tfevents.1528339039.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  16158208 6月   7 10:42 events.out.tfevents.1528339320.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  37822065 6月   7 13:10 events.out.tfevents.1528344840.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  15580677 6月   7 13:18 events.out.tfevents.1528348710.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  16292127 6月   7 13:23 events.out.tfevents.1528348860.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  24053813 6月   7 17:14 events.out.tfevents.1528361564.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  17893916 6月   8 08:40 events.out.tfevents.1528418212.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  45646638 6月   8 09:58 events.out.tfevents.1528418558.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  59457203 6月   8 11:58 events.out.tfevents.1528423520.vulgatepc*
-rwxrwxrwx 1 vulgate vulgate  14852195 6月   8 10:05 graph.pbtxt*
-rwxrwxrwx 1 vulgate vulgate 438600352 6月   8 11:15 model.ckpt-16928.data-00000-of-00001*
-rwxrwxrwx 1 vulgate vulgate     40511 6月   8 11:15 model.ckpt-16928.index*
-rwxrwxrwx 1 vulgate vulgate   8285444 6月   8 11:15 model.ckpt-16928.meta*
-rwxrwxrwx 1 vulgate vulgate 438600352 6月   8 11:25 model.ckpt-18139.data-00000-of-00001*
-rwxrwxrwx 1 vulgate vulgate     40511 6月   8 11:25 model.ckpt-18139.index*
-rwxrwxrwx 1 vulgate vulgate   8285444 6月   8 11:25 model.ckpt-18139.meta*
-rwxrwxrwx 1 vulgate vulgate 438600352 6月   8 11:35 model.ckpt-19353.data-00000-of-00001*
-rwxrwxrwx 1 vulgate vulgate     40511 6月   8 11:35 model.ckpt-19353.index*
-rwxrwxrwx 1 vulgate vulgate   8285444 6月   8 11:35 model.ckpt-19353.meta*
-rwxrwxrwx 1 vulgate vulgate 438600352 6月   8 11:45 model.ckpt-20572.data-00000-of-00001*
-rwxrwxrwx 1 vulgate vulgate     40511 6月   8 11:45 model.ckpt-20572.index*
-rwxrwxrwx 1 vulgate vulgate   8285444 6月   8 11:45 model.ckpt-20572.meta*
-rwxrwxrwx 1 vulgate vulgate 438600352 6月   8 11:55 model.ckpt-21784.data-00000-of-00001*
-rwxrwxrwx 1 vulgate vulgate     40511 6月   8 11:55 model.ckpt-21784.index*
-rwxrwxrwx 1 vulgate vulgate   8285444 6月   8 11:55 model.ckpt-21784.meta*

 



############start facter RCNN config#######################
# Faster R-CNN with Resnet-101 (v1) configured for the Oxford-IIIT Pet Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
  faster_rcnn {
    num_classes: 2
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }
    feature_extractor {
      type: 'faster_rcnn_resnet101'
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
        
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
  batch_size: 1
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0003
          schedule {
            step: 0
            learning_rate: .0003
          }
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  ######################################修改了下面这个这个 modify blow ######################################
  #fine_tune_checkpoint: "model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "train.record"
  }
  label_map_path: "annotations/label_map.pbtxt"
}

eval_config: {
  num_examples: 2000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "val.record"
  }
  label_map_path: "annotations/label_map.pbtxt"
  shuffle: false
  num_readers: 1
}


############end facter RCNN config#######################

<img src=screenshots/starwars_small.gif width=100% />

# Custom Object Detection with TensorFlow
Object detection allows for the recognition, detection, and localization of multiple objects within an image. It provides us a much better understanding of an image as a whole as apposed to just visual recognition.

**Why Object Detection?**
![](https://cdn-images-1.medium.com/max/1600/1*uCdxGFAuHpEwCmZ3iOIUaw.png)

## Installation

First, with python and pip installed, install the scripts requirements:

```bash
pip install -r requirements.txt
```
Then you must compile the Protobuf libraries:

```bash
protoc object_detection/protos/*.proto --python_out=.
```

Add `models` and `models/slim` to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

>_**Note:** This must be ran every time you open terminal, or added to your `~/.bashrc` file._


## Usage
### 1) Create the TensorFlow Records
Run the script:

```bash
python object_detection/create_tf_record.py
```

Once the script finishes running, you will end up with a `train.record` and a `val.record` file. This is what we will use to train the model.

### 2) Download a Base Model
Training an object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we’ll take an object detector trained on a different dataset, and reuse some of it’s parameters to initialize our new model.

You can find models to download from this [model zoo](https://github.com/bourdakos1/Custom-Object-Detection/blob/master/object_detection/g3doc/detection_model_zoo.md). Each model varies in accuracy and speed. I used `faster_rcnn_resnet101_coco` for the demo.

Extract the files and move all the `model.ckpt` to our models directory.

>_**Note:** If you don't use `faster_rcnn_resnet101_coco`, replace `faster_rcnn_resnet101.config` with the corresponding [config file](https://github.com/bourdakos1/Custom-Object-Detection/tree/master/object_detection/samples/configs)._

### 3) Train the Model
Run the following script to train the model:

```bash
python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config
```

### 4) Export the Inference Graph
When you model is ready depends on your training data, the more data, the more steps you’ll need. My model was pretty solid at ~4.5k steps. Then, at about ~20k steps, it peaked. I even went on and trained it for 200k steps, but it didn’t get any better.

>_**Note:** If training takes way to long, [read this](https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e)._

I recommend testing your model every ~5k steps to make sure you’re on the right path.

You can find checkpoints for your model in `Custom-Object-Detection/train`.

Move the model.ckpt files with the highest number to the root of the repo:
- `model.ckpt-STEP_NUMBER.data-00000-of-00001`
- `model.ckpt-STEP_NUMBER.index`
- `model.ckpt-STEP_NUMBER.meta`

In order to use the model, you first need to convert the checkpoint files (`model.ckpt-STEP_NUMBER.*`) into a frozen inference graph by running this command:

```bash
python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix model.ckpt-STEP_NUMBER \
        --output_directory output_inference_graph
```

You should see a new `output_inference_graph` directory with a `frozen_inference_graph.pb` file.

### 5) Test the Model
Just run the following command:

```bash
python object_detection/object_detection_runner.py
```

It will run your object detection model found at `output_inference_graph/frozen_inference_graph.pb` on all the images in the `test_images` directory and output the results in the `output/test_images` directory.

## Results
Here’s what I got from running my model over all the frames in this clip from Star Wars: The Force Awakens.

[![Watch the video](screenshots/youtube.png)](https://www.youtube.com/watch?v=xW2hpkoaIiM)

## License

[MIT](LICENSE)
