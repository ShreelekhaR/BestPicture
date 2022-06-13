import cv2

# read the image
img = cv2.imread("face_angle_2.jpg")

# show the image
cv2.imshow(winname="Face", mat=img)

# Wait for a key press to exit
cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()