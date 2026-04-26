
import asyncio
import binascii
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def camera_to_niryo_coords(viewport_coords):
    """ Converts overhead camera viewport coords to niryo table-top coords.

    Viewport (0,0) -> Bottom Left, Viewport (1,1) -> Top right.
    """ 

    table_top_z = 0.63
    m_y = 1
    c_x = -0.5
    m_x = -1.8
    c_y = -0.5
    viewport_x = viewport_coords[0]
    viewport_y = viewport_coords[1]
    return (
        -m_y * (viewport_y + c_y),
        m_x * (viewport_x + c_x),
        table_top_z
    )

def frame_to_tensor(frame):
    """Converts a camera frame message to an image tensor (RGB)"""
    
    height = frame["frame"]["height"]
    width = frame["frame"]["width"]
    img_data = list(binascii.a2b_base64(frame["frame"]["data"]))
    
    # remove alpha channel
    img_data = [val for index, val in enumerate(img_data) if (index + 1) % 4 != 0]
    
    # convert to tensor
    img_data = np.array(img_data)
    img_data = np.reshape(img_data, (height, width, 3))
    img_data = tf.convert_to_tensor(img_data, dtype=np.ubyte)
    img_data = tf.image.convert_image_dtype(img_data, tf.uint8)[tf.newaxis, ...]
    
    return img_data

async def tf_load_model(url):
    """Loads a TF-Hub detector model asynchronously."""
    loop = asyncio.get_running_loop()
    detector = await loop.run_in_executor(None, lambda url: hub.load(url), url)
    return detector

async def tf_detect(detector, img):
    """Runs detector inference asynchronously."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None,
        lambda detector, img: detector(img), detector, img)
    return result

def tf_result_to_list(results):
    """Converts an object detection result to a convenient list.

    List items: {class_id, score, bbox}
    Order: score descending.
    """
    result_list = []
    for i in range(0, len(results["detection_classes"][0])):
        res = {}
        res['class_id'] = results["detection_classes"][0][i]
        res['score'] = results["detection_scores"].numpy()[0][i].astype(float)
        res['bbox'] = [
            results["detection_boxes"].numpy()[0][i][0].astype(float),
            results["detection_boxes"].numpy()[0][i][1].astype(float),
            results["detection_boxes"].numpy()[0][i][2].astype(float),
            results["detection_boxes"].numpy()[0][i][3].astype(float)
            ]
        result_list.append(res)
    return result_list


def annotate_camera(boxes, camera="OverheadCameraPlayer_annotations"):
    """Emits sequence to annotate camera with boxes in IDE.
    """
    
    annotation = {'annotateCamera': camera, 'boxes': boxes}
    print("!escape!" + json.dumps(annotation) + "!escape!")