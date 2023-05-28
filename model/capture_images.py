# Import required libraries
import os
import cv2

# Create a folder to store the images
DATA_DIR = "./dataset"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Number of classes to be trained. I'm only training four classes: H, E, L, and O
number_of_classes = 4
class_size = 100  # Number of images to be captured for each class

cap = cv2.VideoCapture(0)  # Capture video from the camera
if not cap.isOpened():
    raise IOError("The camera cannot be opened!")

try:
    for i in range(0, number_of_classes):
        # Create a folder for each class
        class_dir = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        print("Capturing images for class {}.".format(i))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally

            # Display the instructions for capturing images
            text = "Press 'C' to capture images for class {}.".format(i)
            if cv2.waitKey(25) & 0xFF == ord("c"):
                # Change the text to "Capturing..." when the user presses the specified key
                text = "Capturing..."

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Capturing images...", frame)

            if text == "Capturing...":
                break

        for j in range(class_size):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Flip the frame horizontally
                img_name = os.path.join(class_dir, "{}.jpg".format(j))
                cv2.imwrite(img_name, frame)
            else:
                print("Failed to capture image for class {}.".format(i))

        print("Done capturing images for class {}.".format(i))

finally:
    cap.release()
    cv2.destroyAllWindows()
