import cv2
from multiprocessing import Queue, Pool
from queue import LifoQueue
from threading import Thread
import random
import os
import pings
import time
import datetime
import platform
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import multiprocessing
########################################################################################################################
# CONSTANTES

VIOLET_PALE           = "#CC66FF"
font 		          = "Constantia"
fontButtons           = (font, 12)

########################################################################################################################
# parametres de la fonction launch_analysis

input_q = Queue()  # fps is better if queue is higher but then more lags
output_q = Queue()
output_q_label = Queue()
score_threshold = 0.5
max_object_display = 25
txt_thickness = 2
FONT_SIZE = 18

CWD_PATH = os.getcwd()

#####################################################

NUM_CLASSES = 90
# Path to frozen detection graph. This is the actual model that is used for the object detection.

#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_NAME ='ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'


PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'models', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'labelmap_face.pbtxt')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories_all = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

categories_name_list = [dict_cat['name'] for dict_cat in categories_all]
categories_id_list = [dict_cat['id'] for dict_cat in categories_all]

categories_name_list_for_inference = ['humain']
#Uncomment below if you want to use all classes
#categories_name_list_for_inference = categories_name_list

categories_id_list_for_inference = []
for value_name in categories_name_list_for_inference:
    for dict_class in categories_all:
        if dict_class['name'] == value_name:
            value_id = dict_class['id']
    categories_id_list_for_inference.append(value_id)

categories = [{'id': categories_id_list_for_inference[index], 'name': categories_name_list_for_inference[index]} for index in range(0, len(categories_name_list_for_inference))]

category_index = label_map_util.create_category_index(categories)

NUM_CLASSES = len(categories_name_list_for_inference)

def detect_objects(image_np, sess, detection_graph):
    global PATH_TO_FONT
    global FONT_SIZE
    global agnostic_bool

    FONT_NAME = 'Antonio.ttf'
    FONT_NAME = os.path.join(CWD_PATH, 'fonts', FONT_NAME)
    #PATH_TO_FONT = os.path.join(CWD_PATH, 'fonts', FONT_NAME)
    PATH_TO_FONT = FONT_NAME

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    #print("PATH_TO_CKPT = ", PATH_TO_CKPT)
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    #print("######", boxes)
    #print("******", scores)
    #print("++++++", classes)
    #print("-------", image_np, type(image_np))


    # Visualization of the results of a detection.
    table_detect=vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_object_display,
        min_score_thresh=score_threshold,
        agnostic_mode=False,
        line_thickness=txt_thickness,
        fontname=PATH_TO_FONT,
        fontsize=FONT_SIZE
    )

    return image_np, table_detect


def worker(input_q, output_q ,output_q_label):
    global PATH_TO_CKPT
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()

        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:
        #print("worker : PATH TO CKPT : ", PATH_TO_CKPT)
        list_detect=[]
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_detect,table_detect = detect_objects(frame_rgb, sess, detection_graph)
        list_detect.append(frame_rgb_detect)
        list_detect.append(table_detect)
        output_q.put(list_detect)
        output_q_label.put(table_detect)

    sess.close()

if __name__ == '__main__':
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    QUEUE_SIZE = 10

    input_q = Queue(maxsize=QUEUE_SIZE)
    frame_q = Queue(maxsize=3*QUEUE_SIZE)
    output_q = Queue(maxsize=QUEUE_SIZE)
    detect_q = Queue(maxsize=QUEUE_SIZE)

    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q, output_q_label))
        t.daemon = True
        t.start()

    SRC = "videos/vid.mp4"

    video_capture = cv2.VideoCapture(SRC)

    def get_stream_frame():
        while True:
            ret, frame = video_capture.read()
            #print("ret = ", ret, "type de frame = ", type(frame), "isOpened = ", video_capture.isOpened())

            if str(SRC).endswith(".mp4") or str(SRC).endswith(".avi") or str(SRC).endswith(".mkv"):
                frame_q.put(frame)
            else:
                if frame_q.qsize() == 3 * QUEUE_SIZE:
                    trash = frame_q.get()
                frame_q.put_nowait(frame)

    for i in range(1):
        t2 = Thread(target=get_stream_frame)
        t2.daemon = True
        t2.start()

    while True:  # fps._numFrames < 120
        t0 = time.time()
        frame = frame_q.get()
        t = time.time()

        input_q.put(frame)
        start_time = datetime.datetime.now()

        frame_rgb_detect, table_detect = output_q.get()
        output_rgb = cv2.cvtColor(frame_rgb_detect, cv2.COLOR_RGB2BGR)

        #Calcul de performance FPS
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = 1 / elapsed_time

        # Affichage de l'image
        cv2.imshow('Video', output_rgb)
        cv2.imwrite('temp.jpg', output_rgb)

        cv2.waitKey(10)

        print('[PERFORMANCE] Elapsed Time 1: {:.2f}'.format((time.time() - t) * 1000), 'ms')
        print('[PERFORMANCE] Objects Detected: ', str(table_detect))
        print('=========================================================')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    cv2.destroyAllWindows()

