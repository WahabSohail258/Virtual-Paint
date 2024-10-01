import numpy as np
import cv2
import os
import HandTracking as htm
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

drawthickness = 15
eraserthickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

folderpath = "./VirtualPaint"
myList = os.listdir(folderpath)

overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    if image is not None:
        image = cv2.resize(image, (1280, 125))
        overlayList.append(image)
    else:
        print(f"Failed to load image: {impath}")

header = overlayList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)
drawcolor = (255, 0, 255)

recognized_text = ""

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    rescaled = cv2.resize(eroded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return rescaled

while True:
    success, img = cap.read()
    if not success:
        print("Unable to read the frame from the camera!")
        break

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist, _ = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()

        if fingers[3] and fingers[4] and fingers[2] and not fingers[0] and not fingers[1]:
            print("Text recognition mode activated")
            # Save the canvas image
            canvas_image_path = 'canvas_image.png'
            cv2.imwrite(canvas_image_path, imgCanvas)
            
            # Load the saved image and preprocess it for OCR
            canvas_image = cv2.imread(canvas_image_path)
            preprocessed_image = preprocess_for_ocr(canvas_image)

            # Use pytesseract to extract text
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
            recognized_text = text.strip()
            print(f"Recognized Text: {recognized_text}")

        if fingers[1] and not fingers[2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawcolor == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraserthickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, drawthickness)

            xp, yp = x1, y1

        elif fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")

            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawcolor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawcolor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawcolor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawcolor = (0, 0, 0)

    img[0:125, 0:1280] = header

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

