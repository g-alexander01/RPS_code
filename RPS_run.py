from keras.models import Model, load_model
model = load_model("rps4.h5")

# This list will be used to map probabilities to class names, Label names are in alphabetical order.
label_names = ['nothing', 'paper', 'rock', 'scissor']
label_numbers = [0,1,2,3]

import serial
import time
import cv2
import numpy as np

port = "/dev/cu.usbmodem14101"

# change video port to 0 for my computer
cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))
 
while True:
     
    ret, frame = cap.read()
    if not ret:
        break
         
    frame = cv2.flip(frame, 1)
            
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)
         
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)
 
    roi = frame[5: box_size-5 , width-box_size + 5: width -5]
     
    # Normalize the image like we did in the preprocessing step, also convert float64 array.
    roi = np.array([roi]).astype('float64') / 255.0
  
    # Get model's prediction.
    pred = model.predict(roi)
     
    # Get the index of the target class.
    target_index = np.argmax(pred[0])
 
    # Get the probability of the target class
    prob = np.max(pred[0])

    # Show results
    cv2.putText(frame, "prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob*100 ),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Rock Paper Scissors", frame)

    arduino = serial.Serial(port, 9600, timeout=.1)
    time.sleep(1) #give the connection a second to settle
    arduino.write(label_numbers[np.argmax(pred[0])])
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
            
 
cap.release()
cv2.destroyAllWindows()