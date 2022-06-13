import cv2
import dlib
import sys
import copy
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

    return (x_fav,y_fav)

def video_input(fav_points):
    fav_x , fav_y = fav_points
    
    # read the image
    cap = cv2.VideoCapture(0)
    num = 0
    while True:
        num +=1
        _, frame = cap.read()
        copy_frame = copy.deepcopy(frame)
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
            new_x = []
            new_y = []
            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                new_x.append(x)
                new_y.append(y)
                # # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                cv2.circle(img=frame, center=(fav_x[n], fav_y[n]), radius=3, color=(255, 0, 0), thickness=-1)

    # show the image
        cv2.imshow(winname="Face", mat=frame)
      
        
       
        diff = facial_differences(fav_points,(new_x,new_y)) 

        if diff < 3000:
            # show the image
            cv2.imshow(winname="Similar", mat=copy_frame)
            cv2.imwrite('Captured_Image.jpg', copy_frame)
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

def facial_differences(fav_points,new_points):
    fav_x , fav_y = fav_points
    new_x , new_y = new_points
    diff = 0

    # Euclidean distance between two points
    for i in range(0,68):
        diff += (fav_x[i] - new_x[i])**2 + (fav_y[i] - new_y[i])**2
    return diff

    # # cosine similarity between two points
    # for i in range(0,68):
    #     diff += (fav_x[i] - new_x[i])**2 + (fav_y[i] - new_y[i])**2
    # return diff


if __name__ == "__main__":
    args = sys.argv[1:]
    points = get_fav_features(args[0])
    # print(facial_differences(points,points))
    video_input(points)

    