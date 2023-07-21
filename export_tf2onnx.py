
# import the necessary packages
from tensorflow.keras.models import load_model, save_model
import argparse
import tf2onnx
import onnx
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def model2onnx(args):
	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])
	onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

	onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
	onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'

	onnx.save(onnx_model, args['output'])


if __name__ == "__main__":
  wieght_path = 'weights/model.h5'
  if not os.path.exists(wieght_path):
      gdd.download_file_from_google_drive(file_id='1ja_aGYWaAEbzDqMvm-svSCS2FrEy7XA4',
                                          dest_path=wieght_path)

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-m", "--model", type=str,
    default="./weights/model.h5",
    help="path to trained face mask detector model")
  ap.add_argument("-o", "--output", type=str,
    default='./weights/mask_detector.onnx',
    help="path to trained face mask detector model")
  args = vars(ap.parse_args())
  model2onnx(args)