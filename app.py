import face_recognition
import cv2
import os
import numpy 

# Завантажуємо фотографії з папки faces
known_faces = []
known_names = []

for file in os.listdir("faces"):
    image = face_recognition.load_image_file(f"faces/{file}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(file.split(".")[0])

# Ініціалізуємо веб-камеру
cap = cv2.VideoCapture(0)

while True:
    # Захоплюємо кадр з веб-камери
    ret, frame = cap.read()

    # Шукаємо обличчя на кадрі
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Перевіряємо чи збігається хоча б одне обличчя з відомими
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = numpy.argmin(face_distances)

        # Якщо є збіг, то виводимо ім'я та відсоток схожості
        if matches[best_match_index]:
            name = known_names[best_match_index]
            similarity = (1 - face_distances[best_match_index]) * 100
            name += f" ({similarity:.2f}%)"

        # Вирисовуємо рамку навколо обличчя та виводимо ім'я
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Відображаємо кадр
    cv2.imshow("Face Recognition", frame)

    # Натисніть 'q' для виходу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершуємо роботу
cap.release()
cv2.destroyAllWindows()