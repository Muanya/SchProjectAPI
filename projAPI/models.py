import base64
import os
import cv2
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from SchProjectAPI.settings import BASE_DIR

"""
Emotion detection model processing
"""

EMOTION_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

MODEL_RELATIVE_PATH = 'projAPI/trained_model/projectModel.h5'

SAVED_MODEL = os.path.join(BASE_DIR, MODEL_RELATIVE_PATH)

CASCADE_FILE = 'projAPI/haarcascades/haarcascade_frontalface_alt.xml'
CASCADE_FILE_PATH = os.path.join(BASE_DIR, CASCADE_FILE)


def is_base64(sb):
    try:
        if isinstance(sb, str):
            # If there's any unicode here, an exception will be thrown and the function will return false
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False


class EmotionLoader:
    """
    Handles the detection of emotion of a base64 encoded image

    """
    model = None
    image = None

    def __init__(self, img):
        assert img is not None, 'Image cannot be none'
        self.image = img
        self.model = keras.models.load_model(SAVED_MODEL)  # load the model

    @staticmethod
    def get_faces(img):
        detector = cv2.CascadeClassifier(CASCADE_FILE_PATH)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        rects = detector.detectMultiScale(img_gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) > 0:
            i = 0
            gap = 10
            results = []
            for rect in rects:
                i += 1
                x, y, w, h = rect
                rgb_img = cv2.putText(rgb_img, str(i), (x - gap, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cropped_img = img_gray[y:y + h, x:x + w]
                cropped_img = cv2.resize(cropped_img, (48, 48))
                test_img = cropped_img[np.newaxis, ...]
                test_img = np.repeat(test_img[..., np.newaxis], 3, -1)
                datagen = ImageDataGenerator(
                    rescale=1.0 / 255.0,
                    featurewise_center=True,
                    featurewise_std_normalization=True
                )
                # Loading the mean and std used in training -- to be reformatted
                datagen.mean = np.array([0.3381997, 0.3381997, 0.3381997], dtype=np.float32).reshape((1, 1, 3))
                datagen.std = np.array([0.23850498, 0.23850498, 0.23850498], dtype=np.float32).reshape((1, 1, 3))
                test_img = datagen.flow(test_img, batch_size=1)
                results.append(test_img.next())
            return rgb_img, results
        return None, None

    """
    This method decodes the emotion of the image stored in @img
    """

    def decode_image(self):
        # decode the image from base64
        image_64_decode = base64.b64decode(self.image)
        npImg = np.frombuffer(image_64_decode, dtype=np.uint8)
        img = cv2.imdecode(npImg, 1)

        # get all the faces in the image
        labelled_img, faces = self.get_faces(img)
        result = {}
        if faces is not None:
            result_arr = []
            for face in faces:
                emotion_dict = {}
                prediction = self.model(face)
                prediction = prediction.numpy().squeeze().tolist()
                ls = sorted(prediction, reverse=True)
                for i in range(3):
                    ind = prediction.index(ls[i])
                    emotion = EMOTION_DICT[ind]
                    emotion_dict[emotion] = ls[i]
                result_arr.append(emotion_dict)

            _, buffer = cv2.imencode('.JPEG', labelled_img)
            buffer = base64.b64encode(buffer)
            # result['image_encoded'] = buffer
            result['emotions'] = result_arr
        else:
            result['info'] = 'No face found in the image'
            result['emotions'] = [{}]

        return result


if __name__ == '__main__':
    image = open('testImages/image1.jpg', 'rb')
    image_read = image.read()
    image_64_encode = base64.encodebytes(image_read)
    x = EmotionLoader(image_64_encode).decode_image()
    print(x['emotions'])


def print_emotions(frame):
    _, img_frame = cv2.imencode('.jpg', frame)
    img_frame = base64.b64encode(img_frame)
    x = EmotionLoader(img_frame).decode_image()
    print(x)

    new_image = frame
    emotion = x['emotions'][0]
    row = 100

    for key in emotion:
        new_image = cv2.putText(
            img=new_image,
            text="{} => {}".format(key, emotion[key]),
            org=(100, row),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(125, 246, 55),
            thickness=1
        )

        row = row + 100

    cv2.imshow("Detected emotion", new_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected emotion")

# if __name__ == "__main__":
#     vid = cv2.VideoCapture(0)
#
#     while vid.isOpened():
#         _, frame = vid.read()
#
#         cv2.imshow('webcam', frame)
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             print_emotions(frame)
#
#         if cv2.waitKey(25) & 0xFF == ord('z'):
#             break
#
#     vid.release()
#     cv2.destroyAllWindows()
