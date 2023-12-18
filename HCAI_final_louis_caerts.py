"""
This python program uses Mediapipe's facial landmarks detection model to count and record the times of blinking/yawning.
Commentary has been provided throughout the entirety of this file to guide the reader throughout the workings of the code.

@author: Louis Caerts
@date: 21.12.2023
"""


###########################################################################################
###                                                                                     ###
###                                   STEP 1: IMPORTS                                   ###
###                                                                                     ###
###########################################################################################

# Mediapipe's facial landmarking
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Data manipulations/operations
import math
import numpy as np

# Video playing
import cv2

# Blink/yawn data visualization
import cvzone
from cvzone.PlotModule import LivePlot

# CSV file manipulation
import csv



###########################################################################################
###                                                                                     ###
###                                  STEP 2: FUNCTIONS                                  ###
###                                                                                     ###
###########################################################################################

def draw_landmarks_on_image(rgb_image, detection_result):
  """
  Draws the facial landmarks found on the current frame so the user can see how accurate the model was.
  This is an example function provided by Mediapipe's example on how to visualize the results of their facial landmarking.
  It is not strictly needed for this program but makes it easier to see the blink/yawn detection in action.

  Source:
    https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb

  Args:
    rgb_image: The image object that contains the current frame of the video being played.
    detection_result: The result of Mediapipe's facial landmarking.

  Returns:
    annotated_image: A replica of rbg_image with the facial landmarks drawn overtop.
  """
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_iris_connections_style())

  return annotated_image

def end_of_vid(vidcap):
  """
  Checks if the end of the video file has been reached.

  Args:
    vidcap: The VideoCapture object that contains the video being played.

  Returns:
    type(bool): Whether or not the end of the video has been reached.
  """
  
  if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
    return True
  return False

def found_face(faceLandmarkResult):
  """
  Checks if a FaceLandMarkResult object (from mediapipe) is empty or not.
  In order words, this function checks whether Mediapipe managed to detect a face in the current frame.

  Args:
    faceLandMarkResult: The result of Mediapipe's facial landmarking.

  Returns:
    type(bool): Whether or not a face was found in the FaceLandMarkResult object.
  """

  resultEmpty = len(faceLandmarkResult.face_landmarks) == 0
  return not resultEmpty

def add_to_data(data, count, input):
  """
  Adds the timecodes corresponding to blink or yawn start-/end-times to the corresponding list holding the rest.
  This function is used in preparation to print all blinking- and yawning-data to .csv files at the end of the program.

  Args:
    data: The list holding the other blink/yawn timecodes.
    count: The blink/yawn count, used to know at which index to add the data point.
    input: A list of length 2 holding respectively the start- and end-timecode of the blink or yawn.
  """

  index = count - 1

  if index == len(data):
    data.append(['',''])
    data[index][0] = input[0]
    data[index][1] = input[1]
  elif index > len(data):
    print("WARNING: Attempted to add data in a way that would leave empty rows.")
    return
  elif index < len(data):
    print("WARNING: Attempted to overwrite previously written data.")
    return
    
def vidpos_to_timecode(framerate, frameid):
  """
  Calculates the timecode of the current position in the video.

  Args:
    framerate: The framerate of the video.
    frameid: The number of the current frame in the video.

  Returns:
    timecode: The timecode as a string, corresponding to the following format: <hours>:<minutes>:<seconds>:<frames>.
  """

  msecs = (1000 / framerate) * frameid                                # The milliseconds passed since the beginning of the video.
  frames = math.floor(frameid - math.floor(msecs/1000) * framerate)   # The frame number of the current second (frames < framerate)

  hours = math.floor(msecs / 3600000)
  msecs = msecs - hours * 3600000
  minutes = math.floor(msecs / 60000)
  msecs = msecs - minutes * 60000
  seconds = math.floor(msecs / 1000)
  msecs = msecs - seconds * 1000

  timecode = "{:02d}".format(hours) + ":" +\
    "{:02d}".format(minutes) + ":" +\
    "{:02d}".format(seconds) + ":" +\
    "{:02d}".format(frames)
  
  return timecode



def create_csv_file(name):
  """
  Creates a csv file to write blinking and yawning timecodes to.
  Fails if a file of the same name already exists.

  Args:
    name: The name of the file one wants created.
  """

  try:
    f = open(name + ".csv", "w", newline='')
    writer = csv.writer(f)
    header = ['in', 'out']
    writer.writerow(header)
    f.close()
  except FileExistsError:
    print(name + ".csv already exists.")
    exit(0)



###########################################################################################
###                                                                                     ###
###                                   STEP 3: GLOBALS                                   ###
###                                                                                     ###
###########################################################################################

# Objects needed to perform facial landmark detection
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
  base_options = BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
  running_mode=VisionRunningMode.VIDEO,
  num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Variables needed to check for blinking/yawning
rightEyeTrackingPoints = [159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160]
mouthTrackingPoints = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
ratioListRightEye = []
ratioListMouth = []
blinking = False
yawning = False
blinkThreshold = 220
yawnThreshold = 450

# Variables needed to save blinking/yawning data
blinkCounter = 0
yawnCounter = 0
blinkData = []
yawnData = []
currentBlink = ['','']
currentYawn = ['','']

# Variables/objects needed to display the progress of the program
colorRightEye = (255, 0, 255)
colorMouth = (255, 0, 255)
rightEyePlotY = LivePlot(640, 360, [0, blinkThreshold*2], invert=True)
mouthPlotY = LivePlot(640, 360, [0, yawnThreshold*2], invert=True)



###########################################################################################
###                                                                                     ###
###                                 STEP 4: PREPARATION                                 ###
###                                                                                     ###
###########################################################################################

# Open the video file
videoName = 'P028-3-final-cropped'
cap = cv2.VideoCapture(videoName + ".mp4")
if not cap.isOpened():
  print(f'\033[91m{"ERROR: " + videoName + ".mp4 was not found."}\033[0m')
  exit(0)

# Get the framerate (used in calculating the timecodes)
framerate = cap.get(cv2.CAP_PROP_FPS)

# Create the CSV files to write blinking/yawning data to
create_csv_file(videoName + '-blink-plot')
blinkFile = open(videoName + '-blink-plot.csv', 'w', newline='')
blinkWriter = csv.writer(blinkFile)
create_csv_file(videoName + '-yawn-plot')
yawnFile = open(videoName + '-yawn-plot.csv', 'w', newline='')
yawnWriter = csv.writer(yawnFile)



###########################################################################################
###                                                                                     ###
###                                  STEP 5: MAIN CODE                                  ###
###                                                                                     ###
###########################################################################################

while not end_of_vid(cap):

  # Read the next frame, transform it to a suitable image format and perform the facial landmarking
  success, img = cap.read()
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
  face_landmarker_result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

  
  if found_face(face_landmarker_result):

    # Draw the facial landmarking results over the current frame and save the landmarks in a separate variable
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)
    face = face_landmarker_result.face_landmarks


    ###################################
    ###       BLINK DETECTION       ###
    ###################################

    # Get the key landmarks of the right eye and transform them to the video's dimensions
    upRightEye = [face[0][159].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][159].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    downRightEye = [face[0][145].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][145].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    leftRightEye = [face[0][133].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][133].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    rightRightEye = [face[0][33].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][33].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    
    # Calculate the distance between the right eye's vertical and horizontal key landmarks
    lengthVerRightEye = math.sqrt((upRightEye[0] - downRightEye[0]) ** 2 + (upRightEye[1] - downRightEye[1]) ** 2)
    lengthHorRightEye = math.sqrt((rightRightEye[0] - leftRightEye[0]) ** 2 + (rightRightEye[1] - leftRightEye[1]) ** 2)
    
    # Calculate the eye openness ratio and add it to the queue
    ratioRightEye = int((lengthVerRightEye / lengthHorRightEye) * 1000)
    ratioListRightEye.append(ratioRightEye)
    if len(ratioListRightEye) > 2:
      ratioListRightEye.pop(0)

    # Use the mean of the past few eye openness ratios to stabilize the value and write it to file
    ratioAvgRightEye = sum(ratioListRightEye) / len(ratioListRightEye)
    blinkWriter.writerow([ratioAvgRightEye,])

    # Check if the target is blinking and update the blinking timecodes/plot accordingly
    if ratioAvgRightEye < blinkThreshold:
      if not blinking:
        currentBlink[0] = vidpos_to_timecode(framerate, cap.get(cv2.CAP_PROP_POS_FRAMES))
        blinkCounter += 1
        colorRightEye = (0,200,0)
        blinking = True
    else:
      if blinking:
        currentBlink[1] = vidpos_to_timecode(framerate, cap.get(cv2.CAP_PROP_POS_FRAMES))
        add_to_data(blinkData, blinkCounter, currentBlink)
        blinking = False
        colorRightEye = (255, 0, 255)
        
        
    ##################################
    ###       YAWN DETECTION       ###
    ##################################

    # Entirely analogous to the blinking section above
    upMouth = [face[0][13].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][13].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    downMouth = [face[0][14].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][14].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    leftMouth = [face[0][308].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][308].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    rightMouth = [face[0][78].x * cap.get(cv2.CAP_PROP_FRAME_WIDTH), face[0][78].y * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]

    lengthVerMouth = math.sqrt((upMouth[0] - downMouth[0]) ** 2 + (upMouth[1] - downMouth[1]) ** 2)
    lengthHorMouth = math.sqrt((leftMouth[0] - rightMouth[0]) ** 2 + (leftMouth[1] - rightMouth[1]) ** 2)

    ratioMouth = int((lengthVerMouth / lengthHorMouth) * 1000)
    ratioListMouth.append(ratioMouth)
    if len(ratioListMouth) > 5:
        ratioListMouth.pop(0)

    ratioAvgMouth = sum(ratioListMouth) / len(ratioListMouth)
    yawnWriter.writerow([ratioAvgMouth,])

    if ratioAvgMouth > yawnThreshold:
      if not yawning:
        currentYawn[0] = vidpos_to_timecode(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_POS_FRAMES))
        yawnCounter += 1
        colorMouth = (0,200,0)
        yawning = True
    else:
      if yawning:
        currentYawn[1] = vidpos_to_timecode(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_POS_FRAMES))
        add_to_data(yawnData, yawnCounter, currentYawn)
        yawning = False
        colorMouth = (255, 0, 255)
        
        
    ###################################
    ###        VISUALIZATION        ###
    ###################################

    # Visualize the blink/yawn counts
    cvzone.putTextRect(annotated_image, f'Blink Count: {blinkCounter}', (50, 100), colorR=colorRightEye)
    cvzone.putTextRect(annotated_image, f'Yawn Count: {yawnCounter}', (50, 200), colorR=colorMouth)

    # Visualize the eye/mouth openness
    rightEyeImgPlot = rightEyePlotY.update(ratioAvgRightEye, colorRightEye)
    mouthImgPlot = mouthPlotY.update(ratioAvgMouth, colorMouth)

    # Visualize the plot titles
    cvzone.putTextRect(rightEyeImgPlot, f'Right Eye', (10, 50), scale=2, thickness=2, colorR=colorRightEye)
    cvzone.putTextRect(mouthImgPlot, f'Mouth', (10, 50), scale=2, thickness=2, colorR=colorMouth)

    # Organize the various visual elements
    cv2img = cv2.resize(annotated_image, (640, 360))
    cv2imgStack = cvzone.stackImages([rightEyeImgPlot, cv2img, mouthImgPlot], 2, 1)

  else:
    # Organize the various visual elements
    cv2img = cv2.resize(img, (640, 360))
    cv2imgStack = cvzone.stackImages([cv2img, cv2img], 2, 1)
        
    # Write data to the plot files indicating that no face was found during this frame
    blinkWriter.writerow([1000.0])
    yawnWriter.writerow([0.0])

  # Show the results of the visualizations
  cv2.imshow("Image", cv2imgStack)
  cv2.waitKey(10)



###########################################################################################
###                                                                                     ###
###                                 STEP 6: TERMINATION                                 ###
###                                                                                     ###
###########################################################################################

# Close the files holding the continues blinking/yawning data (the plot's data)
blinkFile.close()
yawnFile.close()

# Open, write, and close the files holding the individual blink timecodes
create_csv_file(videoName + '-blink-timecodes')
f = open(videoName + '-blink-timecodes.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerows(blinkData)
f.close()

# Open, write, and close the files holding the individual blink timecodes
create_csv_file(videoName + '-yawn-timecodes')
f = open(videoName + '-yawn-timecodes.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerows(yawnData)
f.close()