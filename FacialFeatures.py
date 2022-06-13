import cv2
import dlib
import sys
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_fav_features(impath):
        # read the image
    img = cv2.imread(impath)

    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        x_fav = []
        y_fav = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            x_fav.append(x)
            y_fav.append(y)
            # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        
    # show the image
    cv2.imshow(winname="Face", mat=img)

    # Delay between every fram
    cv2.waitKey(delay=0)

    # Close all windows
    cv2.destroyAllWindows()

    return (x,y)

def video_input():
    # read the image
    cap = cv2.VideoCapture(0)
    num = 0
    while True:
        num +=1
        _, frame = cap.read()
        
        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # show the image
        cv2.imshow(winname="Face", mat=frame)

        if num == 100:
            # show the image
            cv2.imshow(winname="Similar", mat=frame)
            cv2.imwrite('Captured_Image.jpg', frame)
            # Wait for a key press to exit
            if cv2.waitKey(delay=0):
                cv2.destroyWindow("Similar")

        # Exit when escape is pressed
        if cv2.waitKey(delay=1) == 27:
            break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = sys.argv[1:]
    get_fav_features(args[0])
    video_input()