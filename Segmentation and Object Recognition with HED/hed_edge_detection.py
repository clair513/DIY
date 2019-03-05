# Loading required libraries:
import cv2

# PATH to pre-trained model files:
protoFile = '~/Desktop/hed/model/deploy.prototxt'
weightsFile = '~/Desktop/hed/model/hed_pretrained.caffemodel'

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


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
cv2.dnn_registerLayer('Crop', CropLayer)

img = cv2.imread('~/Desktop/hed/images/Input_cr7.jpg')
H, W = img.shape[:2]

img_blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
net.setInput(img_blob)
out = net.forward()
out = cv2.resize(out[0, 0], (W, H))
out = (255 * out).astype('uint8')

cv2.imwrite('~/Desktop/hed/images/Output_hedImage_cr7.jpg', out)
cv2.imshow('HED Image Output', out)
cv2.waitKey(0)
