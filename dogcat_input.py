# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import isfile, join

from os import listdir

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original dogcat
# image . If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 300

# Global constants describing the dogcat data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10

# ==fcnotes==called in distorted_inputs and inputs
def read_dogcat(input_queue):
  """Reads and parses examples from dogcat data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class DOGCATRecord(object):
    pass
  result = DOGCATRecord()

  label = input_queue[1]

  label = tf.reshape(label,[1])

  result.height = 300
  result.width = 300
  result.depth = 3
  # image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  # record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the dogcat format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # reader = tf.WholeFileReader()
  # ==fcnotes== Returns the next record (key, value) pair produced by a reader.
  file_contents = tf.read_file(input_queue[0])
  # result.key, value = reader.read(input_queue[0])
  result.key=input_queue[0]
  record_bytes = tf.image.decode_jpeg(file_contents, channels=3)

  resized_record_bytes = tf.image.resize_images(record_bytes,(IMAGE_SIZE,IMAGE_SIZE),method=0)


  # Convert from a string to a vector of uint8 that is record_bytes long.
  # ==fcnotes== why overwrite record_bytes?
  #record_bytes = tf.decode_raw(value, tf.uint8)

  # label = label_by_name(result.key)
  #label=1

  # label_tf=tf.convert_to_tensor([label])

  result.label = tf.cast(label,tf.int32)

  # The first bytes represent the label, which we convert from uint8->int32.
  # result.label = tf.cast(
  #     tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # depth_major = tf.reshape(record_bytes, [result.depth, result.height, result.width])
  # # Convert from [depth, height, width] to [height, width, depth].
  # result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  result.uint8image = resized_record_bytes

  return result

# ==fcnotes==called in distorted_inputs and inputs method in current file
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def label_by_name(name):
    if(name.startswith("dog")):
        return 1
    else:
        return 0

# not called in current file
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for dogcat training using the Reader ops.

  Args:
    data_dir: Path to the dogcat data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  onlyfiles = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.jpg')) ]
  filepaths = [join(data_dir, f) for f in onlyfiles]
  labels = [label_by_name(f) for f in onlyfiles]
  # Create a queue that produces the filenames to read.
  # filenames_queue = tf.train.string_input_producer(filepaths)
  # labels_queue = tf.train.string_input_producer(labels)
  images_tensor = tf.convert_to_tensor(filepaths, dtype=tf.string)
  filename_tensor = tf.convert_to_tensor(onlyfiles, dtype=tf.string)
  labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([filepaths, labels],shuffle=False)

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_dogcat(input_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d cat and dog images before starting to train. '
           'This will take a few minutes...' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def predict_input_get_resized_image(path):
    file_contents = tf.read_file(path)
    # result.key, value = reader.read(input_queue[0])
    record_bytes = tf.image.decode_jpeg(file_contents, channels=3)
    resized_images = tf.image.resize_images(record_bytes, (IMAGE_SIZE, IMAGE_SIZE), method=0)
    float_image = tf.image.per_image_standardization(resized_images)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return float_image

def inputs(eval_data, data_dir, batch_size):
  """Construct input for dogcat evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the dogcat data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    onlyfiles = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.jpg')) ]
    filepaths = [join(data_dir, f) for f in onlyfiles]
    labels = [label_by_name(f) for f in onlyfiles]
    # Create a queue that produces the filenames to read.
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    onlyfiles = [f for f in listdir(data_dir+"/test") if (isfile(join(data_dir+"/test", f)) and f.endswith('.jpg')) ]
    filepaths = [join(data_dir+"/test", f) for f in onlyfiles]
    labels = [label_by_name(f) for f in onlyfiles]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    input_queue = tf.train.slice_input_producer([filepaths, labels],shuffle=False)

    # Read examples from files in the filename queue.
    read_input = read_dogcat(input_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
