import tensorflow as tf
import cv2
image=cv2.imread('/home/ubuntu/Desktop/170927_064540055_Camera_6.jpg')
std_image=tf.image.per_image_standardization(image)
with tf.Session() as sess:
    result=sess.run(std_image)
    cv2.imsave('/home/ubuntu/Desktop/result1.jpg',result)