import cv2
import os

input_path = "dataset"
output_path = "faces"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

for person in os.listdir(input_path):

    person_input = os.path.join(input_path, person)
    person_output = os.path.join(output_path, person)

    os.makedirs(person_output, exist_ok=True)

    # Skip if already processed
    if len(os.listdir(person_output)) > 0:
        print(person, "faces already detected! Skipping...")
        continue

    count = 0

    for file in os.listdir(person_input):

        img_path = os.path.join(person_input, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(120, 120)
        )

        if len(faces) == 0:
            continue
        
        # Keeps largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        faces = faces[:1]

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            
            # Resize for CNN
            face = cv2.resize(face, (100, 100))

            save_path = os.path.join(person_output, f"{count + 1}.jpg")
            
            cv2.imwrite(save_path, face)
            cv2.imshow("Face", face)
            
            key = cv2.waitKey(100)  # display for 100ms
            
            if key == 27:  # If escape key is pressed, exit early
                break

            count += 1

    print(person, "faces detected and saved:", count)

cv2.destroyAllWindows()
print("All faces detected and saved successfully!")