# Import the required libraries
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model.model import Model

# Load the model
model = Model()
model.load_state_dict(torch.load('model/model.pt'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("The camera cannot be opened!")

# Create a named window for the cropped hand image
cv2.namedWindow('Cropped Hand', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Create a copy of the frame
    image = frame.copy()

    # Convert the image to the YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Apply adaptive thresholding on the Cr channel to segment the hand region
    _, cr_thres = cv2.threshold(
        ycrcb[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cr_thres = cv2.morphologyEx(
        cr_thres, cv2.MORPH_OPEN, kernel, iterations=2)
    cr_thres = cv2.dilate(cr_thres, kernel, iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        cr_thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assuming it's the hand)
    contour = max(contours, key=cv2.contourArea)

    # Skip the frame if the contour is too small
    if cv2.contourArea(contour) < 30000:
        # Display the original frame without any modifications
        cv2.imshow('ASL Recognition', frame)

        # Display a black image instead of the cropped hand
        cv2.imshow('Cropped Hand', np.zeros((224, 224, 3)))

    else:
        # Create a mask of the hand region
        mask = np.zeros(cr_thres.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Find the bounding box of the hand region
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the hand region from the original image
        cropped_hand = masked_image[y:y+h, x:x+w]

        # Resize the cropped hand to 224x224
        cropped_hand = cv2.resize(cropped_hand, (224, 224))

        # Convert the cropped hand to PIL Image
        cropped_hand = Image.fromarray(cropped_hand)

        # Preprocess the cropped hand
        input_tensor = transform(cropped_hand).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()

        # Map the predicted class index to its corresponding label
        class_labels = ['H', 'E', 'L', 'O']
        predicted_label = class_labels[predicted_class]

        # Draw the bounding box around the hand region
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 0, 255), thickness=3)

        # Display the predicted class on the image
        cv2.putText(frame, predicted_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('ASL Recognition', frame)

        # Display the cropped hand image in a separate window
        cropped_hand_np = np.array(cropped_hand)
        cv2.imshow('Cropped Hand', cropped_hand_np)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
