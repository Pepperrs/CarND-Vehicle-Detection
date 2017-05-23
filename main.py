import sys
import cv2
import cars
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from calibration import calibration, generate_warp_config
import lanelines


def setup():

    global ret, mtx, dist, rvecs, tvecs
    ret, mtx, dist, rvecs, tvecs = calibration()
    print("Generated calibration data!")

    global warp_matrix, warp_matrix_inverse
    warp_matrix, warp_matrix_inverse = generate_warp_config()


def process_single_image(image_location, i = 1):
    test_image = mpimg.imread(image_location)
    test_image = process_frame(test_image)
    mpimg.imsave("test_image" + str(i) + ".jpg", test_image)


def process_video(video_location):
    video = VideoFileClip(video_location)
    video_processed = video.fl_image(process_frame)
    video_processed.write_videofile("project_output.mp4", audio=False)


def preprocess_frame(image):

    # Distortion Correction
    image = cv2.undistort(image, mtx, dist, None, mtx)

    return image


def process_frame(image):
    original = image

    # run preprocessing (distortion correction)
    # at the moment only required for lane lines
    # image = preprocess_frame(image)

    # run line detection
    #image = lanelines.detect(image, original, warp_matrix, warp_matrix_inverse)

    # run car detection
    image = cars.detect(image)

    return image





def main(argv):

    # setup()
    cars.setup()
    cars.train()



    for i in range(1, 6):
        process_single_image("test_images/test" + str(i) + ".jpg", i)

    #process_video("project_video.mp4")




if __name__ == "__main__":
    main(sys.argv)


