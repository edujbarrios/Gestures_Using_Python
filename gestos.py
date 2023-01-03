import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Funcion para saber si es la mano derecha o no
def rightHand(thumb_x,pinky_x):
  if(thumb_x < pinky_x): # Comprueba la distancia x del pulgar al indice 
    return True
  return False

def thumbOpenned(thumb_tip,thumb_bottom,pinky_x):
  if(rightHand(thumb_tip,pinky_x)):       # Si es mano derecha
    return (thumb_tip < thumb_bottom)    
  else:                                   # Si es mano izquierda
    return not (thumb_tip < thumb_bottom)

# Comprueba que el pulgar este mirando hacia arriba y los dedos esten recogidos en un puño
def thumb_up(lm):
  return ((lm[2].y > lm[4].y) and (lm[7].y < lm[11].y) and (lm[11].y < lm[15].y) and (lm[15].y < lm[19].y)  
            and (abs(lm[5].x - lm[6].x) < 0.05)  and (abs(lm[5].y - lm[6].y) < 0.05) and (abs(lm[4].x - lm[5].x) > 0.05)) 

# Comprueba que el pulgar este mirando hacia abajo y los dedos esten recogidos en un puño
def thumb_down(lm):
  return ((lm[2].y < lm[4].y) and (lm[7].y > lm[11].y) and (lm[11].y > lm[15].y) and (lm[15].y > lm[19].y)
            and (abs(lm[5].x - lm[6].x) < 0.05)  and (abs(lm[5].y - lm[6].y) < 0.05) and (abs(lm[4].x - lm[5].x) > 0.05)) 

# Comprueba que los dedos indice y corazon esten juntos, el corazon este separado del anular y el anular este junto al meñique
def long_life(lm):
  return abs(lm[12].x-lm[8].x) < 0.08 and abs(lm[20].x-lm[16].x) < 0.08 and (abs(lm[16].x-lm[12].x) >= 0.08)

# Comprueba que el indice este separado del corazon 
def peace(lm):
  return (lm[8].y < lm[6].y) and (lm[12].y < lm[10].y) and (abs(lm[12].x-lm[8].x) >= 0.08)
 



cap = cv2.VideoCapture(0)
coords_index = []
edit_mode = False
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=3) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #height, width and depth (RGB=3) of image
    (h,w,d) = image.shape
  
    if results.multi_hand_landmarks:
      hand_index = 0

      for hand_landmarks in results.multi_hand_landmarks:
        count = 0
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      #----------------
      # Detect fingers
      #----------------
      
      # list of finger tips locators, 4 is thumb, 20 is pinky finger
        tipIds = [4, 8, 12, 16, 20]
      
        lm = hand_landmarks.landmark
      
      # x,y coordinates of pinky tip. Coordinates are normalized to [0.0,1.0] with width and height of the image
        lm[tipIds[4]].x
        lm[tipIds[4]].y

      
      # CONTAR LOS DEDOS LEVANTADOS
        if(thumbOpenned(lm[tipIds[0]].x, lm[tipIds[0]-2].x, lm[tipIds[4]].x)): 
            count = count + 1

        for i in range(1,len(tipIds)):
          if( lm[tipIds[i]].y < lm[tipIds[i]-2].y): 
            count = count + 1                
      # ACTIVAR MODO DIBUJO CON EL PULGAR HACIA ARRIBA
        if((edit_mode == False) and thumb_up(lm)): 
          edit_mode = True
      # ACTIVAR MODO DIBUJO CON EL PULGAR HACIA ABAJO
        elif((edit_mode == True) and thumb_down(lm)): 
          edit_mode = False
          coords_index.clear()
      # PINTAR CON EL DEDO ÍNDICE
        if (edit_mode):
          coords_index.append((int(lm[tipIds[1]].x*w),int(lm[tipIds[1]].y*h)))
          for i in range(0,len(coords_index)):
            cv2.circle(image, coords_index[i], 10, (0,0,255), -1)
      # SÍMBOLO DE LARGA VIDA
        if (count == 5):
          if (long_life(lm)): 
            cv2.putText(image, "Larga vida", (int(0.1*w),int(0.1*h)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 3)
      # SÍMBOLO DE LA PAZ
        if (count == 2):
          if(peace(lm)):
            cv2.putText(image, "Paz", (int(0.1*w),int(0.1*h)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 3)
            

      # Contador de dedos en el centro de la mano
        cv2.putText(image, "{}".format(count), (int(lm[0].x*w),int(lm[0].y*h)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0), thickness = 5)
      

    cv2.imshow('MediaPipe Hands', image)    
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
      
cap.release()
