from yolo.utils.draw import (rectangle, get_boxes)
from yolo.yolo.util import *


def detection(image_loader, model, args, draw=True, path=None, detector=None):
    _class_names = ''

    if detector == 'cars':
        _class_names = args.cars_names

    if detector == 'alpr':
        _class_names = args.alpr_names

    if detector == 'characters':
        _class_names = args.characters_names

    classes = load_classes(_class_names)
    image_processed = image_loader[0].cuda() if args.cuda else image_loader[0]
    # Original images from batches
    image_original = image_loader[1]
    # Dimension of original image
    height = model.net_info['height']
    # Get predictions
    predictions = predict(model, image_processed, classes, args)
    if type(predictions) == int:
        return 0
    if args.cuda:
        torch.cuda.synchronize()
    # Predictions rescaled to the original image
    scaled_predictions = rescale_prediction(image_loader, predictions, height, args)
    # Save image in output folder
    if draw:
        rectangle(scaled_predictions, image_original, classes, args, path)
    # Get boxes from detected objects
    boxes = get_boxes(predictions)
    return boxes


def handle_dimensions(image_loader, predictions, cuda):
    """
    Repeat image dimensions along axis=1, twice and get cuda version
        if cuda is available
    :param image_loader: loader of an image
    :param cuda: cuda.is_avaible()
    :return: image dimensions of class torch FloatTensor or cuda version of it
    """
    image_dimensions = image_loader[2]
    image_dimensions = torch.FloatTensor(image_dimensions).repeat(1, 2)
    image_dimensions = image_dimensions.cuda() if cuda else image_dimensions
    image_dimensions = torch.index_select(image_dimensions, 0, predictions[:, 0].long())
    return image_dimensions


def predict(model, image_processed, classes, args):
    """
    Get predictions from model
    :param model: network model
    :param image_processed: preprocessed image
    :param classes: classes for detection
    :param args: program parameters
    :return: predictions
    """
    num_classes = len(classes)
    with torch.no_grad():
        predictions = model(Variable(image_processed), args.cuda)
    predictions = write_results(predictions, args.confidence, num_classes, nms=True, nms_conf=args.nms_thresh)
    return predictions


def rescale_prediction(image_loader, predictions, height, args):
    """
    Rescale predictions to get boxes according to original size of image
    :param image_loader: loader of images
    :param predictions: predictions from network
    :param height: height of input image
    :param args: program argumetns
    :return: rescaled predictions
    """
    image_dimensions = handle_dimensions(image_loader, predictions, args.cuda)
    scaling_factor = torch.min(height / image_dimensions, 1)[0].view(-1, 1)
    predictions[:, [1, 3]] -= (height - scaling_factor * image_dimensions[:, 0].view(-1, 1)) / 2
    predictions[:, [2, 4]] -= (height - scaling_factor * image_dimensions[:, 1].view(-1, 1)) / 2

    predictions[:, 1:5] /= scaling_factor

    for i in range(predictions.shape[0]):
        predictions[i, [1, 3]] = torch.clamp(predictions[i, [1, 3]], 0.0, image_dimensions[i, 0])
        predictions[i, [2, 4]] = torch.clamp(predictions[i, [2, 4]], 0.0, image_dimensions[i, 1])

    return predictions
