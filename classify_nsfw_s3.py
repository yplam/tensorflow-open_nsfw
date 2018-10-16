#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
import boto3
import tempfile
import os

import numpy as np

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


def main(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument("path", help="Path to the input images. Only jpeg images are supported.")

  parser.add_argument("-b", "--bucket", required=True,
                      help="AWS S3 bucket name")

  parser.add_argument("-m", "--model_weights", required=True,
                      help="Path to trained model weights file")

  parser.add_argument("-o", "--output", required=True,
                      help="Path to output result file")

  parser.add_argument("-l", "--image_loader",
                      default=IMAGE_LOADER_YAHOO,
                      help="image loading mechanism",
                      choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

  parser.add_argument("-t", "--input_type",
                      default=InputType.TENSOR.name.lower(),
                      help="input type",
                      choices=[InputType.TENSOR.name.lower(),
                               InputType.BASE64_JPEG.name.lower()])

  args = parser.parse_args()

  s3 = boto3.client('s3')
  bucket_name = ''
  for bucket in s3.list_buckets().get('Buckets'):
    if bucket.get('Name') == args.bucket:
      bucket_name = bucket.get('Name')

  if not bucket_name:
    print("Bucket {} not available".format(args.bucket))
    exit(-1)

  images = []
  next_token = ''
  while True:
    if next_token:
      response = s3.list_objects_v2(
        Bucket=bucket_name,
        Delimiter='|',
        EncodingType='url',
        MaxKeys=1000,
        Prefix=args.path,
        ContinuationToken=next_token,
        FetchOwner=False
      )
    else:
      response = s3.list_objects_v2(
        Bucket=bucket_name,
        Delimiter='|',
        EncodingType='url',
        MaxKeys=1000,
        Prefix=args.path,
        FetchOwner=False
      )
    content = response.get('Contents')
    next_token = response.get('NextContinuationToken')
    for item in content:
      images.append(item.get('Key'))
    if not next_token:
      break
    print(next_token)
    # if len(images) > 100:
    #   break

  model = OpenNsfwModel()

  with tf.Session() as sess:

    input_type = InputType[args.input_type.upper()]
    model.build(weights_path=args.model_weights, input_type=input_type)

    fn_load_image = None

    if input_type == InputType.TENSOR:
      if args.image_loader == IMAGE_LOADER_TENSORFLOW:
        fn_load_image = create_tensorflow_image_loader(sess)
      else:
        fn_load_image = create_yahoo_image_loader()
    elif input_type == InputType.BASE64_JPEG:
      import base64
      fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

    sess.run(tf.global_variables_initializer())

    output = open(args.output, "a")

    for item in images:
      temp_file = tempfile.mkstemp()
      s3.download_file(bucket_name, item, temp_file[1])
      try:
        image = fn_load_image(temp_file[1])
      except IOError:
        print("Read Image Error")
        pass
      predictions = sess.run(model.predictions, feed_dict={model.input: image})
      output.write("https://www.themebeta.com/media/cache/400x225/files/{}, {}\r\n".format(item, predictions[0][0]))
      os.remove(temp_file[1])
      print("Results for https://www.themebeta.com/media/cache/400x225/files/{} : {}".format(item, predictions[0][0]))

    output.close()

if __name__ == "__main__":
  main(sys.argv)
