import cv2
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Преобразуйте кадр в черно-белый формат
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Преобразуйте черно-белое изображение в трехмерный массив с тремя каналами
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Передайте BGR изображение в Mediapipe для обработки
            results = hands.process(image=bgr)

            # Визуализируйте результаты
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Отобразите кадр с результатами на экране
            cv2.imshow('Hand Tracking', frame)

            # Выход из цикла при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
