import cv2
import mediapipe as mp
from models.adaline_sgd import AdalineSGD
from vision.angle_utils import angle
from data.training_data import xtr, ytr

ph = mp.solutions.pose
pose = ph.Pose(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#nrainiw fel model(training the model )
clf = AdalineSGD(eta=0.01, n_iter=50)
clf.fit(xtr, ytr)

cap = cv2.VideoCapture(0)
count = 0
previous_position = -1
#supposina eli initialement -down-(we supposed that the initial position is downn)

while True:
    r, f = cap.read()
    #ken r false yaani l'image mat9ratch(image non lue)
    if not r:
        break

    i = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    results = pose.process(i)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        left_sh = [lm[11].x, lm[11].y]
        right_sh = [lm[12].x, lm[12].y]
        shoulder_mid = [(left_sh[0] + right_sh[0]) / 2,(left_sh[1] + right_sh[1]) / 2]

        left_hip = [lm[23].x, lm[23].y]
        right_hip = [lm[24].x, lm[24].y]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2,(left_hip[1] + right_hip[1]) / 2]

        left_heel = [lm[29].x, lm[29].y]
        right_heel = [lm[30].x, lm[30].y]
        feet_mid = [(left_heel[0] + right_heel[0]) / 2,(left_heel[1] + right_heel[1]) / 2]

        left_elbow = [lm[13].x, lm[13].y]
        right_elbow = [lm[14].x, lm[14].y]

        left_wrist = [lm[15].x, lm[15].y]
        right_wrist = [lm[16].x, lm[16].y]

        left_elbow_angle = angle(left_sh, left_elbow, left_wrist)
        right_elbow_angle = angle(right_sh, right_elbow, right_wrist)
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        shoulder_angle = angle(left_hip, shoulder_mid, right_sh)
        trunk_angle = angle(shoulder_mid, hip_mid, feet_mid)

        features = [elbow_angle, shoulder_angle, trunk_angle]

        prediction = clf.predict([features])[0]

        if previous_position == -1 and prediction == 1:
            count += 1

        previous_position = prediction

        mp_drawing.draw_landmarks(f,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Push-Up Counter", f)
    """L if loutaniya tekhdem ken ki tenzel 3al q fel clavier """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
