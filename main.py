import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

model = tf.keras.models.load_model("path/sign_language_model1.h5")

input_shape = model.input_shape
print(f"Model expects input shape: {input_shape}")

class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z", "Nothing", "Space", "Delete"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

recent_predictions = []
prediction_confidence_threshold = 0.4  

prev_frame_time = 0
new_frame_time = 0

cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Debug View", 224, 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    valid_hand_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, c = frame.shape
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx, lmy = int(lm.x * w), int(lm.y * h)
                landmarks.append([lmx, lmy])
            
            landmarks = np.array(landmarks)
            x, y, width, height = cv2.boundingRect(landmarks)
            
            center_x, center_y = x + width // 2, y + height // 2
            size = max(width, height)
            padding = int(size * 0.2)  
            
            x1 = max(0, center_x - size//2 - padding)
            y1 = max(0, center_y - size//2 - padding)
            x2 = min(w, center_x + size//2 + padding)
            y2 = min(h, center_y + size//2 + padding)
            
            hand_roi = frame[y1:y2, x1:x2]
            
            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                valid_hand_detected = True
                
                hand_roi_debug = hand_roi.copy()
                
                hand_roi_resized = cv2.resize(hand_roi, (224, 224)) 
                
                hand_roi_normalized = hand_roi_resized / 255.0
                
                # Option 2: Try converting to grayscale if your model was trained on grayscale
                # hand_roi_gray = cv2.cvtColor(hand_roi_resized, cv2.COLOR_BGR2GRAY)
                # hand_roi_gray = np.expand_dims(hand_roi_gray, axis=-1)  # Add channel dimension
                # hand_roi_normalized = hand_roi_gray / 255.0
                
                # Option 3: Using standard preprocessing used by some pretrained models
                # hand_roi_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(hand_roi_resized)
                
                debug_view = (hand_roi_normalized * 255).astype(np.uint8)
                cv2.imshow("Debug View", debug_view)
                
                hand_roi_expanded = np.expand_dims(hand_roi_normalized, axis=0)
                
                predictions = model.predict(hand_roi_expanded)
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]
                
                if confidence > prediction_confidence_threshold:
                    recent_predictions.append(predicted_class)
                    if len(recent_predictions) > 10: 
                        recent_predictions.pop(0)
                    
                    from collections import Counter
                    if recent_predictions:
                        most_common = Counter(recent_predictions).most_common(1)[0][0]
                        
                        label = class_labels[most_common]
                        confidence = predictions[0][most_common] * 100
                        
                        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Low confidence: {confidence:.2f}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if not valid_hand_detected:
        cv2.putText(frame, "No valid hand detected", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Sign Language Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        pass

cap.release()
cv2.destroyAllWindows()
hands.close()