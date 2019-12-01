import argparse

import torch


def _add_detection_config(arg_parser):
    """Adds default configuration parameters to argument parser
        Args:
           arg_parser - argument parser to add parameters
    """
    arg_parser.add_argument("--image", dest='image',
                            help="Image to perform detection upon",
                            default="result.jpg", type=str)
    arg_parser.add_argument("--video", dest='video',
                            help="video to perform detection upon",
                            default="data/video/test.mp4", type=str)
    arg_parser.add_argument("--out_folder", dest='out_folder',
                            help="Image / Directory to store detections to", default="results", type=str)
    arg_parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=1)
    arg_parser.add_argument("--confidence", dest="confidence", type=float,
                            help="Object Confidence to filter predictions", default=0.2)
    arg_parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.2, type=float)

    arg_parser.add_argument("--cars_cfg", dest='cars_cfg',
                            help="Config file of cars", default="yolo/configs/cars.cfg", type=str)
    arg_parser.add_argument("--alpr_cfg", dest='alpr_cfg',
                            help="Config file", default="yolo/configs/alpr.cfg", type=str)
    arg_parser.add_argument("--characters_cfg", dest='characters_cfg',
                            help="Config file", default="yolo/cfg/yolov3-tiny-football.cfg", type=str)

    arg_parser.add_argument("--cars_weights", dest='cars_weights', help="weightsfile",
                            default="yolo/weights/cars.weights", type=str)
    arg_parser.add_argument("--alpr_weights", dest='alpr_weights', help="weightsfile",
                            default="yolo/weights/alpr.weights", type=str)
    arg_parser.add_argument("--characters_weights", dest='characters_weights', help="weightsfile",
                            default="yolo/weights/characters.weights", type=str)

    arg_parser.add_argument("--cars_names", dest='cars_names', help="classes file",
                            default="yolo/configs/coco.names", type=str)
    arg_parser.add_argument("--alpr_names", dest='alpr_names', help="classes file",
                            default="yolo/configs/alpr.names", type=str)
    arg_parser.add_argument("--characters_names", dest='characters_names', help="classes file",
                            default="yolo/configs/characters.names", type=str)

    arg_parser.add_argument("--resolution", dest='resolution',
                            help="Input resolution of the network. Increase to increase accuracy. "
                                 "Decrease to increase speed",
                            default="416", type=str)
    arg_parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                            default="0,1,2", type=str)
    cuda = int(torch.cuda.is_available())
    arg_parser.add_argument("--gpu", dest="cuda", default=cuda, help="If cuda available", type=int)


def init_detection_config():
    """Initializes default configuration for features vector search
        Returns:
            flags - configuration parameters
    """
    arg_parser = argparse.ArgumentParser('Image search service')
    _add_detection_config(arg_parser)
    flags, _ = arg_parser.parse_known_args()

    return flags


def init_config():
    """Initializes configuration parameters container
        Returns:
            flags - training and evaluation configuration parameters
    """
    # Network architecture
    arg_parser = argparse.ArgumentParser('Image search standalone')
    _add_detection_config(arg_parser)
    flags, _ = arg_parser.parse_known_args()

    return flags
