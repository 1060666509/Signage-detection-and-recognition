# Imports Modules
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sort_contours(contours, method="left-to-right"):
    '''
    Sort the contours by some method, acending order.
    Args:
        contours: A list of contours
        method: Sorting method, default as left to right
    
    Return:
        None
    '''
    if method == "left-to-right":
        length = len(contours)
        # bubble sort the contours according to x value, ascendingly
        for i in range(length):
            for j in range(length-i-1):
                x1, _, _, _ = cv2.boundingRect(contours[j])
                x2, _, _, _ = cv2.boundingRect(contours[j+1])
                if x2 < x1:
                    # swap
                    temp = contours[j+1]
                    contours[j+1] = contours[j]
                    contours[j] = temp
    elif method == "top-to-down":
        length = len(contours)
        # bubble sort the contours according to y value, ascendingly
        for i in range(length):
            for j in range(length-i-1):
                _, y1, _, _ = cv2.boundingRect(contours[j])
                _, y2, _, _ = cv2.boundingRect(contours[j+1])
                if y2 < y1:
                    # swap
                    temp = contours[j+1]
                    contours[j+1] = contours[j]
                    contours[j] = temp
    
    return None


def img_processing(img):
    '''
    Given an image, do Gaussian blurring, Morphology transformation, adaptive thresholding, 
    and using Canny edge detector to get an edge map.
    Args:
        img: The original image

    Return:
        img_edge: The resulting edge map 
    '''
    # dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    # apply Gaussion Blur
    img_blur = cv2.GaussianBlur(img_dilate, (3, 3), 0)
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    # get an edge map using Canny edge detector
    img_edge = cv2.Canny(img_th, 100, 200, 255)
    # blur again
    img_edge = cv2.GaussianBlur(img_edge, (1, 1), 0)

    return img_edge


def find_select_contours(img, img_edge):
    '''
    Given an binarized image, find contours and filter the contours, discarding those not satisfiable
    Args:
        img: The original image
        img_edge: The edge image
    Return:
        contour_opt: The best contour selected, ideally it shoulbe be the region for multiple digits
    '''
    # find contours
    contours = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # filter contours
    candidate_list = []
    candidate_list_append = []
    for c in contours:
        # get the bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        # check several things
        # filter by areas of the contour, width-height ratio and y coordinate
        if w*h>3000 and w*h<30000 and w > h*1.3 and w < 2.8*h and y < 500:
            candidate_list_append.append(c)
            # the percentage of extreme high intensity pixels
            percent_white = 1.0*np.sum(img[y:y+h, x:x+w] > 240) / (h*w)
            # the average intensity of the contour
            avg_intensity = 1.0*np.sum(img[y:y+h, x:x+w]) / (h*w)
            # select by high intensity pixel density and average intensity
            if percent_white > 0.00165 and avg_intensity > 30 and avg_intensity < 150:
                candidate_list.append(c)

    # after filtering, select the contour with the maximum area
    if len(candidate_list) > 0:
        areas = [cv2.contourArea(c) for c in candidate_list]
        contour_opt = candidate_list[np.argmax(areas)]
        return contour_opt
    elif len(candidate_list_append) > 0:
        return candidate_list_append[0]
    elif len(contours) > 0:
        return contours[0]
    else:
        return None


def segment_digits(img, contour):
    '''
    Within the contour, segment the digits and return a list of contours for each digits, 
    Ideally the resulting contour list should have length 3 (3 digits)
    Args:
        img: The original image
        contour: The optimal contour return by the previous algorithm steps

    Return:
        img_digits: The segmented image for digits
        digit_contours: A list of contours for digits, ideally should have length 3
    '''
    # get the rectangle info from the contour
    digit_contour_list = []
    if contour is None:
        return digit_contour_list
    x, y, w, h = cv2.boundingRect(contour)
    img_digits = img[y:y+h, x:x+w]
    # apply Gaussion Blur
    img_digits_blur = cv2.GaussianBlur(img_digits, (5, 5), 0)
    # apply Otsu’s binarization methods
    _, img_digits_th = cv2.threshold(img_digits_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # dilate the image
    img_dilate = cv2.dilate(img_digits_th, kernel, iterations=1)
    # find contours for each digits and sort them from left to right
    digit_countours = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for c in digit_countours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w < h and w > 10 and h > 20:
            digit_contour_list.append(c)
            
    # Sort in place
    sort_contours(digit_contour_list, method="left-to-right")

    return img_digits, digit_contour_list


def classify_digits(img_digits, digit_contour_list, classifier):
    '''
    Using the contour image to classify each digits.
    Args:
        img_digits: The segmented digits region of the image
        digit_contour_list: The contour lists return by the previous step
        classifier: The digits and arrow classifier

    Return:
        pred_labels: The predicted digits
    '''
    pred_labels = []
    for i in range(len(digit_contour_list)):
        x, y, w, h = cv2.boundingRect(digit_contour_list[i])
        digit = img_digits[y:y+h, x:x+w]
        # resize into the target resolution (28*40)
        digit_resize = cv2.resize(digit, dsize=(28, 40), interpolation=cv2.INTER_CUBIC)
        # binarize the image
        _, digit_th = cv2.threshold(digit_resize, 128, 255, cv2.THRESH_BINARY)
        # predict using the trained knn model
        _, result, _, _ = classifier.findNearest(digit_th.reshape(1, -1).astype('float32'), k=5)
        pred_label = int(result[0][0])
        pred_labels.append(pred_label)
        # for debugging
        # # plot and save the figure
        # plt.subplot(1, 3, i+1)
        # plt.imshow(digit_resize, cmap="gray")
        # plt.savefig("test.jpg")

    return pred_labels


def train_knn(X_train, Y_train):
    '''
    Using opencv2.ml to create and train a knn classifier
    Args:
        X_train: Training set features, each row is a record
        Y_train: Traininig set labels, each row is label

    Returns:
        model_knn: The trained model
    '''
    model_knn = cv2.ml.KNearest_create()
    model_knn.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

    return model_knn


if __name__ == "__main__":
    # read image
    # train and val are used to test the performance accuary
    #inputdir_name = "/home/student/val/task1/"
    #inputdir_name = "/home/student/train/task1/"
    inputdir_name = "/home/student/test/task1/"
    outputdir_name = "/home/student/shao_weixin_19447557/output/task1/"

    # train a knn
    # prepare training data for the classifier
    Y_train = [0]*5
    # we use number to denote the label for each digit
    for i in range(1, 10):
        Y_train.extend([i]*5)

    file_name_prefix_list = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    im_list = np.array([0]*1120).reshape(1,-1)
    # iterate the directory to load the training data
    for f_prev in file_name_prefix_list:
        for i in range(1,6):
            file_name = f_prev + str(i) + ".jpg"
            # Read as a gray scale image
            im = cv2.imread("/home/student/shao_weixin_19447557/digits_original/" + file_name, 0)
            _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
            im = cv2.resize(im, dsize=(28, 40), interpolation=cv2.INTER_CUBIC)  # resize...
            im_list = np.concatenate((im_list, im.flatten().astype('float32').reshape(1,-1)), axis=0)
        
    im_list = im_list[1:,:]
    X_train = im_list.astype('float32')
    Y_train = np.array(Y_train).astype('float32').reshape(-1,1)
    # train the knn model
    model_knn = train_knn(X_train, Y_train)
    # for the train, change the '11' to '22', 'test' to 'DS', for the val, change 'test' to 'val'
    for i in range(1, 11):
        input_filename = "%stest%02d.jpg" % (inputdir_name, i)
        img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
        # get an edge map
        img_edge = img_processing(img.copy())
        # find contours and get the optimal
        contour = find_select_contours(img, img_edge)
        # segment the image to get digits region
        img_digits, digit_contour_list = segment_digits(img.copy(), contour)

        # save figure
        output_filename = "%sDetectedArea%02d.jpg" % (outputdir_name, i)
        plt.imshow(img_digits, cmap='gray')
        plt.savefig(output_filename)

        # classify the digits
        pred_labels = classify_digits(img_digits, digit_contour_list, model_knn)

        # save the recognition output to a text file
        output_filename = "%sBuilding%02d.txt" % (outputdir_name, i)
        output_str = "Building "
        # printing
        if (len(pred_labels) == 3):
            for i in range(3):
                output_str += str(pred_labels[i])

        with open(output_filename, "w+") as f:
            f.write(output_str)
