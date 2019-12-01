from detection.utils.flags import init_detection_config
from yolo.utils.detector import detection
from yolo.utils.loader import load_model
from yolo.yolo.preprocess import prep_image
import pytesseract
from PIL import Image


import cv2

def detect_cars(image_path):
    """
    Detect car from an image
    :param image_path: string -- path to the image
    :return: list or None -- car coordinates or None
    """
    car_image_loader = prep_image(image_path, car_detector_height, path=True)
    car_results = detection(car_image_loader, car_detector, flags, draw=True, detector='cars')
    car_results = list(filter(lambda result: result[2] in CAR_LABELS, car_results))
    if not car_results == 0:
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
        return alpr_coordinates
    return None


def detect_alpr(_cars_coordinates, _image, draw=True):
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
        alpr_coordinates = detect_alpr_default(_car_coordinates, _image, draw)
        if alpr_coordinates:
            _alprs_coordinates.append(alpr_coordinates)
    return _alprs_coordinates


if __name__ == '__main__':
    INPUT_IMAGE = 'images/1.png'
    # Coco dataset indexes of: Cars, Motorbike, Bus, Truck
    CAR_LABELS = [2, 3, 5, 7]
    # Get default parameters for detection
    flags = init_detection_config()
    # Detector objects
    car_detector = load_model(flags, detector='cars')
    car_detector_height = car_detector.net_info['height']

    alpr_detector = load_model(flags, detector='alpr')
    alpr_detector_height = alpr_detector.net_info['height']

    # Locate the car on the image and get the image back to feed the alpr detector
    cars_coordinates, image = detect_cars(INPUT_IMAGE)
    if None in [cars_coordinates]:
        print('No Cars were found')
    else:
        for cars_coordinate in cars_coordinates:
            car_cords = cars_coordinate
            car_image = image[car_cords[0][1]:car_cords[1][1],car_cords[0][0]:car_cords[1][0]]
            print(cars_coordinate)
            alpr_cords = detect_alpr(cars_coordinate, image, draw=False)
            if None in [alpr_cords]:
                continue
            alpr_cords = alpr_cords[0][0]
            alpr_image = car_image[alpr_cords[0][1]:alpr_cords[1][1],alpr_cords[0][0]:alpr_cords[1][0]]

            pil_image = Image.fromarray(alpr_image)
            config = ("-l eng --oem 3 --psm 7")
            alpr_text = pytesseract.image_to_string(alpr_image, config=config)

            cv2.imshow('car',car_image)
            cv2.imshow(alpr_text,alpr_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()