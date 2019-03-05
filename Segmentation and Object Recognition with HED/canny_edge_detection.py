# Loading required libraries:
import cv2

class CropLayer(object):
    def __init__(self, params, blobs):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]

        self.y_start = (input_shape[2] - target_shape[2]) // 2
        self.x_start = (input_shape[3] - target_shape[3]) // 2
        self.y_end = self.y_start + height
        self.x_end = self.x_start + width

        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y_start:self.y_end, self.x_start:self.x_end]]


cv2.dnn_registerLayer('Crop', CropLayer)

img = cv2.imread('/home/alok/Desktop/hed/images/Input_kiara.jpg')
H, W = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)

cv2.imwrite('/home/alok/Desktop/hed/images/Output_cannyImage_kiara.jpg', canny)
cv2.imshow('Canny Output', canny)
cv2.waitKey(0)
