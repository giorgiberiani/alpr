from yolo.yolo.darknet import Darknet
from yolo.yolo.preprocess import prep_image
import os


def load_model(args, detector=None):
    """
    Load darknet network
    :param model:
    :param args: program parameters
    :return: network model
    """
    _config = ''
    weight_path = ''

    if detector == 'cars':
        _config = args.cars_cfg
        weight_path = args.cars_weights

    if detector == 'alpr':
        _config = args.alpr_cfg
        weight_path = args.alpr_weights
    if detector == 'characters':
        _config = args.characters_cfg
        weight_path = args.characters_weights

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(_config)
    model.load_weights(weight_path)
    print("Network successfully loaded")

    # Set height for model input
    model.net_info["height"] = int(args.resolution)
    assert model.net_info["height"] % 32 == 0
    assert model.net_info["height"] > 32

    # If there's a GPU availible, put the model on GPU
    if args.cuda:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()
    return model


def load_images(args, path=None):
    """
    Load images and create output folder
    :param path:
    :param args: program parameters
    :return: image
    """
    # Detection phase
    if path:
        image = path
    else:
        image = os.path.join(os.getcwd(), args.image)
    image_resolution = int(args.resolution)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    return prep_image(image, image_resolution)




