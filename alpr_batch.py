import os

from detection.utils.flags import init_detection_config
from yolo.utils.detector import detection
from yolo.utils.loader import load_model
from yolo.yolo.preprocess import prep_image
from yolo.utils.draw import (rectangle_alpr, normalize_alpr, plot_images)


def detect_cars(image_path, draw=False):
    """
    Detect car from an image
    :param draw:
    :param image_path: string -- path to the image
    :return: list or None -- car coordinates or None
    """
    car_image_loader = prep_image(image_path, car_detector_height, path=True)
    car_results = detection(car_image_loader, car_detector, flags, draw=draw, detector='cars')
    car_results = list(filter(lambda result: result[2] in CAR_LABELS, car_results))
    if car_results and not car_results == 0:
        return car_results, car_image_loader[1]
    return None, None


def detect_alpr_default(_car_coordinates, _image, draw=True):
    """
    Detect alpr from image and car coordinates
    :param draw:
    :param _car_coordinates: list -- e.g [(x0, y0), (x1, y1), 2]
    :param _image: numpy array -- image array
    :return: list or None -- alpr_coordinates or None if not found
    """
    top_left, bottom_right = _car_coordinates[0:2]
    cropped_image = _image[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
    alpr_image_loader = prep_image(cropped_image, alpr_detector_height, path=False)
    alpr_coordinates = detection(alpr_image_loader, alpr_detector, flags, draw=draw, detector='alpr')
    if not alpr_coordinates == 0:
        alpr_coordinates = normalize_alpr(_car_coordinates, alpr_coordinates)
        return alpr_coordinates
    return None


def detect_alpr(_cars_coordinates, _image, draw=False):
    """
    Detect alpr from many cars on one image
    :param draw:
    :param _cars_coordinates: list -- e.g [[(x0, y0), (x1, y1), 2], [(x0, y0), (x1, y1), 2] ]
    :param _image: numpy array -- image array
    :return:
    """
    if type(_cars_coordinates[0]) == tuple:
        _cars_coordinates = [_cars_coordinates]
    _alprs_coordinates = []
    for _car_coordinates in _cars_coordinates:
        alpr_coordinates = detect_alpr_default(_car_coordinates, _image, draw=draw)
        if alpr_coordinates:
            _alprs_coordinates.append(alpr_coordinates)
    return _alprs_coordinates


def count_mean_detection(_input_folder):
    """
    Count the percentage of ALPR detected on images
    :param _input_folder: string -- path to the folder
    :return: float -- percentage of detection
    """
    _number_of_all_images = 0
    _number_of_positive_images = 0
    for _folder in os.listdir(_input_folder):
        _folder_path = os.path.join(_input_folder, _folder)
        for _image in os.listdir(_folder_path):
            _coordinates = []
            _number_of_all_images += 1
            print('Detecting the image: ', _number_of_all_images)
            _image_path = os.path.join(_folder_path, _image)
            cars_coordinates, image = detect_cars(_image_path, draw=False)
            if None in [cars_coordinates]:
                print('No Cars were found')
            else:
                _coordinates.extend(cars_coordinates)
                alprs_coordinates = detect_alpr(cars_coordinates, image, draw=False)
                if alprs_coordinates:
                    _coordinates.extend(alprs_coordinates[0])
                    _number_of_positive_images += 1
                    rectangle_alpr(_coordinates, image, flags)
    return _number_of_positive_images / _number_of_all_images * 100


if __name__ == '__main__':
    INPUT_FOLDER_PATH = 'test/'
    # Coco dataset indexes of: Cars, Motorbike, Bus, Truck
    CAR_LABELS = [2, 3, 5, 7]
    # Get default parameters for detection
    flags = init_detection_config()
    # Detector objects
    car_detector = load_model(flags, detector='cars')
    car_detector_height = car_detector.net_info['height']

    alpr_detector = load_model(flags, detector='alpr')
    alpr_detector_height = alpr_detector.net_info['height']
    percentage_of_detection = count_mean_detection(INPUT_FOLDER_PATH)
    print('Percentage of detection of ALPR: ', percentage_of_detection)
    plot_images('results/')
