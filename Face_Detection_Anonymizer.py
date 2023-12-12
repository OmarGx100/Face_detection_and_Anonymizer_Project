import cv2
import mediapipe as mp
import os


   # face Detection
# cap = cv2.VideoCapture(0)
# while True :
#     ret, frame = cap.read()
#     mp_face_detection = mp.solutions.face_detection
#     face_detection = mp_face_detection.FaceDetection()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(frame_rgb)
#     if out.detections:
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("face_detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):                    
#         break

# cap.release()                                       
# cv2.destroyAllWindows()



#alittle modification to make it face anonymizer

cap = cv2.VideoCapture(0)
while True :
    ret, frame = cap.read()
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(frame_rgb)
    max_width = frame.shape[1]
    max_height = frame.shape[0]
    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            #ungrapping the bounding box limits
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            # Replace the original region with the blurred one
            if y + h <= max_height: 
                if x + w <= max_width:
                  # Extract the region of interest (ROI)
                  roi = frame[ y : y + h, x : x + w]
                  # Apply Gaussian blur to the ROI
                  blurred_roi = cv2.blur(roi, (50, 50), 0)
                  frame[y  : y + h, x  : x + w] = blurred_roi
                else :
                  # Extract the region of interest (ROI)
                  roi = frame[y  : y + h, x  : max_width]
                  # Apply Gaussian blur to the ROI
                  blurred_roi = cv2.blur(roi, (50, 50), 0)
                  frame[y  : y + h, x  : max_width] = blurred_roi
            else:
                if x + w <= max_width:
                    # Extract the region of interest (ROI)
                    roi = frame[y  : max_height, x  : x + w]
                    # Apply Gaussian blur to the ROI
                    blurred_roi = cv2.blur(roi, (50, 50), 0)
                    frame[y  : max_height, x  : x + w] = blurred_roi
                else :
                     # Extract the region of interest (ROI)
                     roi = frame[y  : max_height, x  : max_width]
                     # Apply Gaussian blur to the ROI
                     blurred_roi = cv2.blur(roi, (50, 50), 0)
                     frame[y  : max_height, x  : max_width] = blurred_roi
    cv2.imshow("face_detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):                    
        break
cap.release()                                       
cv2.destroyAllWindows()




