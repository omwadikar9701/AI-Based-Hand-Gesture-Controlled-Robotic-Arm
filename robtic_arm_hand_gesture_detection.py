import cv2  #OpenCV for image processing and webcam interaction
import mediapipe as mp  #For hand tracking
mp_drawing = mp.solutions.drawing_utils #Drawing hand landmarks and connections
mp_hands = mp.solutions.hands   #for hand tracking

cap = cv2.VideoCapture(0) #to open/caputre image by using camera
with mp_hands.Hands(    #create object as hand
    min_detection_confidence=0.5,   #hyper parameter tuning to get more accurate results
    min_tracking_confidence=0.5) as hands: #Higher values mean more stable tracking but might lose tracking if the hand moves quickly

  while cap.isOpened(): #continues loop till webcam is open
    success, image = cap.read()     #reads frames success-if it is read successfully

    if not success: #if not read frame,
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue #skips to next iterarion
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #1.flips image to horizontal 2.converts BGR(default of opencv) to RBG for mediapipe
     image.flags.writeable = False  #improve performance by telling image data is read only
    results = hands.process(image)   #performs hand traking on image
    # Draw the hand annotations on the image.
    image.flags.writeable = True #set image back to writable so can draw on it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #converts rbg to bgr for opencv
    if results.multi_hand_landmarks:    #check if any hand were detected
      for hand_landmarks in results.multi_hand_landmarks:   #iterates through each detecyed hand 
        mp_drawing.draw_landmarks( #draw hand landmarks and line between joints
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)   #
    cv2.imshow('MediaPipe Hands', image) #to show on screen
    if cv2.waitKey(5) & 0xFF == 27: #time delay for 5 mil sec. if esc key 27 loop breaks
      break
cap.release()   #release webcam
