from flirpy.camera.lepton import Lepton
import numpy as np
import collections
import tensorflow as tf
import cv2
import time
import os, sys
import dlib
from imutils import face_utils
import math

from faceDetection_frozenGraph import TensoflowFaceDector

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/frozen_inference_graph_face.pb'
DLIB_LANDMARKS_MODEL = 'model/shape_predictor_68_face_landmarks.dat'

def to_numpy(self, landmarks):
    coordinates = []
    for i in self.landmark_2D_index:
        coordinates += [[landmarks.part(i).x, landmarks.part(i).y]]
    #print(np.array(coordinates).astype(np.int))
    return np.array(coordinates).astype(np.int)

def main():

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)
    dlib_landmarks = dlib.shape_predictor(DLIB_LANDMARKS_MODEL)

    fov_rgb = 60
    fov_thermal = 71
    fov_offset = 0 #0.055
    fov_correction = math.tan(math.radians(fov_rgb/2))/math.tan(math.radians(fov_thermal/2))
    print(fov_correction)

    # fov_correction = fov_rgb/fov_thermal + fov_offset
    rgb_camera = cv2.VideoCapture(2)
    rgb_camera.set(cv2.CAP_PROP_FPS, 8)
    rgb_fps = rgb_camera.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if (rgb_camera.isOpened() == False): 
      print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(rgb_camera.get(3))
    frame_height = int(rgb_camera.get(4))

    output_vid = cv2.VideoWriter('thermal7.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width*2,frame_height))

    # print("RGB FPS: {:.2f}".format(rgb_fps))

    thermal_camera = Lepton()
    thermal_width, thermal_height = 160, 120
    corrected_thermal_width = thermal_width * fov_correction
    corrected_thermal_height = thermal_height * fov_correction
    interval, count = 0, 0

    cx_thermal, cy_thermal = int(thermal_width/2), int(thermal_height/2)

    radius = 1
    roi = 5

    x1 = cx_thermal - roi
    x2 = cx_thermal + roi
    y1 = cy_thermal - roi
    y2 = cy_thermal + roi


    x1_crop = cx_thermal - int(corrected_thermal_width/2)
    y1_crop = cy_thermal - int(corrected_thermal_height/2)
    x2_crop = cx_thermal + int(corrected_thermal_width/2)
    y2_crop = cy_thermal + int(corrected_thermal_height/2)

    cx_corrected = int((x1_crop + x2_crop)/2)
    cy_corrected = int((y1_crop + y2_crop)/2)

    left_offset = 25 #+20
    right_offset = 25 #-35
    top_offset = 12 #20
    bottom_offset = 12 #-20

    cx_rgb, cy_rgb = int(frame_width/2), int(frame_height/2)

    pts_src = np.array([
                        [179,229],
                        [346, 224],
                        [513, 215],
                        [556, 172],
                        [78, 198]
                        ])
    pts_dst = np.array([
                        [889-640,203],
                        [1018-640, 204],
                        [1145-640, 201],
                        [1175-640, 174],
                        [818-640, 179]
                        ])

    # print('Calculating homographic matrix')
    # H, status = cv2.findHomography(pts_src, pts_dst)


    while True:
        t1 = time.time()

        # ********** THERMAL CAMERA *************************************
        raw_image = thermal_camera.grab(device_id=0)
        # raw_image_corrected = raw_image[y1_crop:y2_crop, x1_crop:x2_crop]
        # thermal_image = ((np.subtract(raw_image_corrected, norm_mat_new.transpose() * 7000)/1500)*255).astype(np.uint8)
        thermal_image = ((np.subtract(raw_image, 7000)/1500)*255).astype(np.uint8)


        #*********** RGB CAMERA PROCESSING *******************************
        ret, rgb_image = rgb_camera.read()
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        if ret == False:
            break
        rgb_height, rgb_width, rgb_channel = rgb_image.shape
        boxes, scores, classes, num_detections = tDetector.run(rgb_image)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        for score, box in zip(scores, boxes):
            if score > 0.5:
                # print('Detecting face ...')
                # ymin, xmin, ymax, xmax = box
                left = int(box[1]*rgb_width)
                top = int(box[0]*rgb_height)
                right = int(box[3]*rgb_width)
                bottom = int(box[2]*rgb_height)

                box_width = right-left
                box_height = bottom-top

                cv2.rectangle(rgb_image, (left, top), (right, bottom), 
                    (255, 255, 255), int(round(rgb_height/150)), 8)
                # cv2.imwrite(filename, image)

                scaled_left     = int((left/rgb_width * corrected_thermal_width) + left_offset)
                scaled_top      = int((top/rgb_height * corrected_thermal_height) + top_offset)
                scaled_right    = int((right/rgb_width * corrected_thermal_width) + right_offset)
                scaled_bottom   = int((bottom/rgb_height * corrected_thermal_height) + bottom_offset)

                # if top > 70 and right:
                #     scaled_left     = int((left+left_offset)/rgb_width * corrected_thermal_width)
                #     scaled_top      = int((top+top_offset)/rgb_height * corrected_thermal_height)
                #     scaled_right    = int((right+right_offset)/rgb_width * corrected_thermal_width)
                #     scaled_bottom   = int((bottom+bottom_offset)/rgb_height * corrected_thermal_height)
                # else:
                #     scaled_left     = int((left+left_offset)/rgb_width * corrected_thermal_width)
                #     scaled_top      = int((top)/rgb_height * corrected_thermal_height)
                #     scaled_right    = int((right+right_offset)/rgb_width * corrected_thermal_width)
                #     scaled_bottom   = int((bottom)/rgb_height * corrected_thermal_height)

                # DLIB FACIAL LANDMARKS
                dlibRect = dlib.rectangle(left, top, right, bottom)
                shape = dlib_landmarks(gray_image, dlibRect)
                # shape = face_utils.shape_to_np(shape)

                # Temperature RoI
                # Case 0: Eye regions
                # leye_x_rgb = shape.part(38).x
                # leye_y_rgb = shape.part(38).y
                # reye_x_rgb = shape.part(40).x
                # reye_y_rgb = shape.part(40).y
                
                # Case 1: Tear Glands regions
                # RGB Image
                ltear_x1_rgb = shape.part(38).x
                ltear_y1_rgb = shape.part(38).y
                ltear_x2_rgb = shape.part(39).x
                ltear_y2_rgb = shape.part(40).y
                cv2.rectangle(rgb_image, (ltear_x1_rgb, ltear_y1_rgb), (ltear_x2_rgb, ltear_y2_rgb), (0,255,0), 2, 2)
                rtear_x1_rgb = shape.part(42).x
                rtear_y1_rgb = shape.part(43).y
                rtear_x2_rgb = shape.part(47).x
                rtear_y2_rgb = shape.part(47).y
                cv2.rectangle(rgb_image, (rtear_x1_rgb, rtear_y1_rgb), (rtear_x2_rgb, rtear_y2_rgb), (0,255,0), 2, 2)
                # Thermal Image
                ltear_x1_thermal = int(shape.part(38).x/rgb_width*corrected_thermal_width+left_offset)
                ltear_y1_thermal = int(shape.part(38).y/rgb_height*corrected_thermal_height+top_offset)
                ltear_x2_thermal = int(shape.part(39).x/rgb_width*corrected_thermal_width+right_offset)
                ltear_y2_thermal = int(shape.part(40).y/rgb_height*corrected_thermal_height+bottom_offset)
                cv2.rectangle(thermal_image, (ltear_x1_thermal, ltear_y1_thermal), (ltear_x2_thermal, ltear_y2_thermal), (0,255,0), 1, 1)
                rtear_x1_thermal = int(shape.part(42).x/rgb_width*corrected_thermal_width+left_offset)
                rtear_y1_thermal = int(shape.part(43).y/rgb_height*corrected_thermal_height+top_offset)
                rtear_x2_thermal = int(shape.part(47).x/rgb_width*corrected_thermal_width+right_offset)
                rtear_y2_thermal = int(shape.part(47).y/rgb_height*corrected_thermal_height+bottom_offset)
                cv2.rectangle(thermal_image, (rtear_x1_thermal, rtear_y1_thermal), (rtear_x2_thermal, rtear_y2_thermal), (0,255,0), 1, 1)

                # Case 2: Inside mouth region
                mouth_x1_rgb = shape.part(48).x
                mouth_y1_rgb = shape.part(50).y
                mouth_x2_rgb = shape.part(54).x
                mouth_y2_rgb = shape.part(57).y
                cv2.rectangle(rgb_image, (mouth_x1_rgb, mouth_y1_rgb), (mouth_x2_rgb, mouth_y2_rgb), (0,255,0), 2, 2)
                # Thermal Image
                mouth_x1_thermal = int(shape.part(48).x/rgb_width*corrected_thermal_width+left_offset)
                mouth_y1_thermal = int(shape.part(50).y/rgb_height*corrected_thermal_height+top_offset)
                mouth_x2_thermal = int(shape.part(54).x/rgb_width*corrected_thermal_width+right_offset)
                mouth_y2_thermal = int(shape.part(57).y/rgb_height*corrected_thermal_height+bottom_offset)
                cv2.rectangle(thermal_image, (mouth_x1_thermal, mouth_y1_thermal), 
                	(mouth_x2_thermal, mouth_y2_thermal), (0,255,0), 1, 1)
                
                # Case 3: Forehead regions
                # forehead_height = shape.part(19).y - top
                # RGB IMAGE
                forehead_x1_rgb = shape.part(19).x
                forehead_y1_rgb = shape.part(19).y
                forehead_x2_rgb = shape.part(24).x
                forehead_y2_rgb = shape.part(24).y
                cv2.rectangle(rgb_image, (forehead_x1_rgb, top), (forehead_x2_rgb, forehead_y2_rgb), (0,255,0), 2, 2)
                # Thermal Image
                forehead_x1_thermal = int(shape.part(19).x/rgb_width*corrected_thermal_width+left_offset)
                forehead_y1_thermal = int(shape.part(19).y/rgb_height*corrected_thermal_height+top_offset)
                forehead_x2_thermal = int(shape.part(24).x/rgb_width*corrected_thermal_width+right_offset)
                forehead_y2_thermal = int(shape.part(24).y/rgb_height*corrected_thermal_height+bottom_offset)
                cv2.rectangle(thermal_image, (forehead_x1_thermal, scaled_top), 
                	(forehead_x2_thermal, forehead_y2_thermal), (0,255,0), 1, 1)


                # cv2.circle(rgb_image, (leye_x_rgb, leye_y_rgb), 1, (0,255,0), 2)
                # cv2.circle(rgb_image, (reye_x_rgb, reye_y_rgb), 1, (0,255,0), 2)

                # leye_x_thermal = int(leye_x_rgb/rgb_width * corrected_thermal_width)
                # leye_y_thermal = int(leye_y_rgb/rgb_height * corrected_thermal_height)
                # reye_x_thermal = int(reye_x_rgb/rgb_width * corrected_thermal_width)
                # reye_y_thermal = int(reye_y_rgb/rgb_height * corrected_thermal_height)

                # cv2.circle(thermal_image, (leye_x_thermal, leye_y_thermal), 1, (0,255,0), -1)
                # cv2.circle(thermal_image, (reye_x_thermal, reye_y_thermal), 1, (0,255,0), -1)


                # for i in range(67):
                # 	# landmarks in rgb image
                #     cv2.circle(rgb_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), 2)
                #     # landmarks in thermal image
                #     cv2.circle(thermal_image, (int((shape.part(i).x)/rgb_width*corrected_thermal_width+left_offset), 
                #     	int((shape.part(i).y)/rgb_height*corrected_thermal_height+top_offset)), 1, (0,0,255), 1)

                    # cv2.imshow('test_image', rgb_image)

                # left_top = H.dot(np.array([[left], [top], [1]]))
                # right_bottom = H.dot(np.array([[right], [bottom], [1]]))

                # x1, y1 = int(left_top[0][0]), int(left_top[1][0])
                # x2, y2 = int(right_bottom[0][0]), int(right_bottom[1][0])


                #************* TEMPERTURE PROCESSING **************************
                # calculate temperature only at RoI
             	# Case 0: Entire Face
                # roi_crop = raw_image[scaled_top:scaled_bottom,scaled_left:scaled_right]
                # Case 1: Mouth inside
                roi_crop = raw_image[mouth_y1_thermal:mouth_y2_thermal,mouth_x1_thermal:mouth_x2_thermal]
                # Case 2: Forehead 
                # roi_crop = raw_image[scaled_top:forehead_y2_thermal,forehead_x1_thermal:forehead_x2_thermal]

                avg_read = np.mean(roi_crop.flatten())
                min_read = np.min(roi_crop.flatten())
                max_read = np.max(roi_crop.flatten())

                # Formula with average reading - (taking all data)-->R2=99.55
                # temp_degF = 0.096156 * avg_read - 675.106714
                # Formula with average reading - (taking selected data)-->R2=99.74%
                temp_degF = 0.09745 * avg_read - 685.80252
                # Formula with maximum reading - (taking selected data)-->R2=99.54%
                # temp_degF = 0.0931129 * max_read - 652.367
                temp_degC = (temp_degF - 32) * 5/9 - 3

                # cv2.rectangle(thermal_image, (x1, y1),(x2, y2),(0, 0, 255), int(round(thermal_height/150)), 1)
                cv2.rectangle(thermal_image, (scaled_left, scaled_top),(scaled_right, scaled_bottom), 
                    (255, 255, 255), int(round(thermal_height/150)), 1)
                cv2.putText(thermal_image, r'Temp: {:.2f} deg C'.format(temp_degC), (scaled_left+2, scaled_top-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)
            # else:
                # print('Not detecting face...')


        t2 = time.time()
        interval += (t2-t1)

        # print('Average temperature: {:.2f}'.format(temp_degF))

        # ************** Usuage: CAMERA CALIBRATION ***************** 
        # if count==0 or interval>= 120:
        #   os.system('spd-say "Next Reading"')
        #   roi_crop = image[y1:y2, x1:x2]
        #   avg_read = np.mean(roi_crop.flatten())
        #   min_read = np.min(roi_crop.flatten())
        #   max_read = np.max(roi_crop.flatten())
        #   # data = roi_crop.flatten().tolist()
        #   # mode_read = max(set(data), key=data.count)
        #   print("Iteration: {}".format(count))
        #   print('Average value: {:.2f}'.format(avg_read))
        #   print('Minimum value: {}'.format(min_read))
        #   print('Maximum value: {}'.format(max_read))
        #   print('---------------------------------')
        #   # print('Mode value: {}'.format(mode_read))
        #   interval = 0
        #   count += 1

        fps = (1/(t2-t1))
        # cv2.circle(rgb_image, (cx_rgb, cy_rgb), 5, (255,0,0), -1)
        # cv2.circle(thermal_image, (cx_corrected, cy_corrected), 1, (0,255,0), -1)
        # cv2.circle(thermal_image, (int(cx_rgb/rgb_width*corrected_thermal_width), 
        # 	int(cy_rgb/rgb_height*corrected_thermal_height)), 1, (255,0,0), -1)


        # print("FPS: {:.2f}".format(fps))
        # cv2.circle(thermal_image, (cx, cy), 1, (0,0,0), -1)
        # cv2.rectangle(thermal_image, (x1,y1), (x2,y2), (255,255,255), 1)
        # cv2.putText(thermal_image, 'FPS: {:.2f}'.format(fps), (3,10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)

        thermal_image = cv2.applyColorMap(thermal_image, cv2.COLORMAP_HOT)
        # print(heatmap.shape)

        disp_window = np.hstack([cv2.resize(rgb_image, (640, 480)), 
                                 cv2.resize(thermal_image, (640, 480))])

        # display rgb image
        cv2.imshow('Temperature Detector', disp_window)
        # cv2.imshow('Temperature Detector1', thermal_image)
        # cv2.imshow('Temperature Detector2', cv2.resize(rgb_image, (640, 480)))
        output_vid.write(disp_window)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
            thermal_camera.close()

    # close the camera
    thermal_camera.close()
    output_vid.release()
    rgb_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
