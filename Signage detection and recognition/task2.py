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
    # sort by x coordiante
    if method == "left-to-right":
        length = len(contours)
        # Bubble sort the contours by x value
        for i in range(length):
            for j in range(length-i-1):
                x1, _, _, _ = cv2.boundingRect(contours[j])
                x2, _, _, _ = cv2.boundingRect(contours[j+1])
                if x2 < x1:
                    # swap
                    temp = contours[j+1]
                    contours[j+1] = contours[j]
                    contours[j] = temp
    # sort by y coordiante
    elif method == "top-to-down":
        length = len(contours)
        # Bubble sort the contours by y value
        for i in range(length):
            for j in range(length-i-1):
                _, y1, _, _ = cv2.boundingRect(contours[j])
                _, y2, _, _ = cv2.boundingRect(contours[j+1])
                if y2 < y1:
                    # swap
                    temp = contours[j+1]
                    contours[j+1] = contours[j]
                    contours[j] = temp
    # sort by area of the rectangle
    elif method == "area":
        length = len(contours)
        # Bubble sort the contours by area
        for i in range(length):
            for j in range(length-i-1):
                area1 = cv2.contourArea(contours[j])
                area2 = cv2.contourArea(contours[j+1])
                if area2 < area1:
                    # swap
                    temp = contours[j+1]
                    contours[j+1] = contours[j]
                    contours[j] = temp

    return None


def img_crop(img):
    '''
    Stage 1 of the whole pipeline. Given an image, do a series of Morphology transformation, Gaussian blurring, adaptive thresholding, 
    and use Canny edge detector to get the desired region: a high rectanglular region containing all digits
    and arrows.
    Args:
        img: The original gray image

    Return:
        img_cropped: The resulting cropped map 
    '''
    # use blackhat operation from Morphology transformation
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
    # apply adaptive thresholding
    img_th = cv2.adaptiveThreshold(img_blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # apply erode operation from Morphology transformation
    img_erose = cv2.erode(img_th,kernel1,iterations = 1)
    # apply close operation from Morphology transformation
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_close = cv2.morphologyEx(img_erose, cv2.MORPH_CLOSE, kernel2)
    # get the edge map
    img_edge = cv2.Canny(img_close, 100, 200, 255)
    # uncommented the following line of code for debugging purpose
    # plt.imshow(im_erose, cmap='gray')

    # find contours and filter the contours
    contours = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    candidate_list = []
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # check several things
        if w*h>1000 and w*h < 10000:
            avg_intensity = 1.0*np.sum(img[y:y+h, x:x+w]) / (h*w)
            if avg_intensity < 120:
                candidate_list.append(c)

    if len(candidate_list) > 4:
        # crop the desired region
        x_vals = [cv2.boundingRect(c)[0] for c in candidate_list]
        y_vals = [cv2.boundingRect(c)[1] for c in candidate_list]
        x_vals_select = []
        y_vals_select = []
        # filter and select from the list
        for i in range(len(x_vals)):
            x_mask1 = [x > x_vals[i]-70 for x in x_vals]
            x_mask2 = [x < x_vals[i]+100 for x in x_vals]
            if not(sum(x_mask1) < 3 or sum(x_mask2) < 3):
                x_vals_select.append(x_vals[i])
                y_vals_select.append(y_vals[i])
        if len(x_vals_select) > 0:
            x_min, x_max = areas = min(x_vals_select), max(x_vals_select)
            y_min, y_max = areas = min(y_vals_select), max(y_vals_select)
        else:
            # do not crop
            x_min, x_max = 0, img.shape[1]
            y_min, y_max = 0, img.shape[0]
    else:
        # do not crop
        x_min, x_max = 0, img.shape[1]
        y_min, y_max = 0, img.shape[0]

    # select the region from the original map and get the cropped image
    x_min_selected, x_max_selected = max(0, x_min-30), min(img.shape[1], x_max+90)
    y_min_selected, y_max_selected = max(0, y_min-30), min(img.shape[0], y_max+100)
    img_cropped = img[y_min_selected:y_max_selected, x_min_selected:x_max_selected]

    return img_cropped


def find_select_contours_signage(img):
    '''
    Given an imagea and its edge map (after processed), find contours and filter the contours to get
    top 6 candidates (if there are more than 6), otherwise, take all of them.
    Args:
        img: The source image (Expected to be the edge map here)
    Return:
        contour_list: The best contour selected, ideally it shoulbe be the region for multiple digits
    '''
    # find contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # filter contours
    candidate_list = []
    for c in contours:
        # get the bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        # check several things
        # filter by areas of the contour and width-height ratio
        if w*h>1000 and w*h<10000 and w > 1.3*h and w < 10*h:
            # select by high intensity pixel density and average intensity
            candidate_list.append(c)

    # after filtering, select the contour with the maximum area
    if len(candidate_list) > 6:
        # pick top 6 candidate with the maximum areas
        sort_contours(candidate_list, method="area")
        candidate_list = candidate_list[:6]
    
    # sort top to down
    sort_contours(candidate_list, method = "top-to-down")
    return candidate_list


def segment_signage(img):
    '''
    Stage 2 of the whole pipeline. Given a (cropped) image, do a series of Morphology transformation, 
    Gaussian blurring, adaptive thresholding, and use Canny edge detector to segment each signage.
    Args:
        img: Expected to be the cropped iamge.

    Return:
        signage_contour_list: A list of contours for digits, ideally should have length 6, may vary in some cases
    '''
    # here a rectangle kernel is important
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    # do the blackhat operation from Morphology transformation, reverse the color
    img_blackhat = cv2.morphologyEx(255-img, cv2.MORPH_BLACKHAT, kernel1)
    # do the close operation from Morphology transformation
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_close = cv2.morphologyEx(img_blackhat, cv2.MORPH_CLOSE, kernel2)
    # apply a blackhat operation again to further reduce environment noise
    # must use the rectanglular kernel
    img_blackhat2 = cv2.morphologyEx(255-img_close, cv2.MORPH_BLACKHAT, kernel1)
    # first apply a Sobel filter on the resulting image, to highlight edges
    # only take derivative along x-axis is sufficient
    grad_x = cv2.Sobel(img_blackhat2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1, delta=0, borderType=cv2.BORDER_REFLECT)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = grad_x.astype("uint8")
    # apply a Gaussion blurring and a close operation on the derivative map
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel1)
    # use OTSU's thresholding method to binarize the gradient map
    _, img_th = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # apply a close operation from Morphology transformation
    img_th = cv2.dilate(img_th, None, iterations=1)
    img_th = cv2.erode(img_th, None, iterations=1)
    # apply an open operation from Morphology transformation
    img_th = cv2.erode(img_th, None, iterations=1)
    img_th = cv2.dilate(img_th, None, iterations=2) # changed
    # apply Canny's edge detector to get an edge map
    img_edge = cv2.Canny(img_th, 100, 200, 255)

    # finally find and filter contours
    signage_contour_list = find_select_contours_signage(img_edge)
    # sort in place
    sort_contours(signage_contour_list, method="top-to-down")

    return signage_contour_list


def segment_digits(img, contour):
    '''
    Within the contour, segment the digits and return a list of contours for each digits, 
    Ideally the resulting contour list should have length 4 (3 digits and 1 arrow)
    Args:
        img: The original image (Expected to be the cropped image)
        contour: The optimal contour return by the previous algorithm steps

    Return:
        img_digits: The segmented image for digits
        digit_contour_list: A list of contours for digits, ideally should have length 4
    '''
    # get the rectangle info from the contour
    digit_contour_list = []
    if len(contour) == 0:
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
        if w > 5 and h > 5:
            digit_contour_list.append(c)
            if len(digit_contour_list) == 4:
                break
            
    # sort in place
    sort_contours(digit_contour_list, method="left-to-right")

    return img_digits, digit_contour_list


def train_knn(X_train, Y_train):
    '''
    Use cv2.ml module to create and train a knn classifier
    Args:
        X_train: Training set features, each row is a record
        Y_train: Traininig set labels, each row is label

    Returns:
        model_knn: The trained model
    '''
    model_knn = cv2.ml.KNearest_create()
    model_knn.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

    return model_knn


def load_and_train_digit_classifier():
    '''
    Load the digit dataset and train a knn classifier on it.
    Args:

    Return:
        model_knn: The KNN digit classifier
    '''
    # prepare training data for the digits classifier
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
            # im = cv2.imread("../data/digits_original/" + file_name, 0)
            _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
            im = cv2.resize(im, dsize=(28, 40), interpolation=cv2.INTER_CUBIC)   # resize...
            im_list = np.concatenate((im_list, im.flatten().astype('float32').reshape(1,-1)), axis=0)
        
    im_list = im_list[1:,:]
    X_train = im_list.astype('float32')
    Y_train = np.array(Y_train).astype('float32').reshape(-1,1)
    # train the knn model
    model_knn = train_knn(X_train, Y_train)

    return model_knn


def load_and_train_arrow_classifier():
    '''
    Load the arrow dataset and train a knn classifier on it.
    Args:

    Return:
        model_knn: The KNN arrow classifier
    '''
    # prepare training data for the digits classifier
    Y_train = [0]*5
    Y_train.extend([1]*5)

    file_name_prefix_list = ["LeftArrow", "RightArrow"]
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

    return model_knn


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
        # for debugging purpose
        # plot and save the figure
        # if len(digit_contour_list) == 1:
        #     plt.imshow(digit_resize, cmap="gray")
        #     plt.savefig("test.jpg")

    return pred_labels


if __name__ == "__main__":
    # three-stage algorithm
    # first stage: detect the black rectangle and crop the original image
    # second stage: segment the crop image to separate groups of digits
    # third stage: within each region after segmentation, segment digits and arrows and recognizet each

    # read in an image
    # inputdir_name = "/home/student/test/task2/"
    # outputdir_name = "/home/student/name/output/task2/"
    #inputdir_name = "/home/student/val/task2/"
    #inputdir_name = "/home/student/train/task2/"
    inputdir_name = "/home/student/test/task2/"
    outputdir_name = "/home/student/shao_weixin_19447557/output/task2/"

    # train two models to recognize digits and arrow separately
    knn_model_digit = load_and_train_digit_classifier()
    knn_model_arrow = load_and_train_arrow_classifier()


    # for each image
     # for the train, change the '11' to '22', 'test' to 'DS', for the val, change 'test' to 'val'
    for i in range(1, 11):
        input_filename = "%stest%02d.jpg" % (inputdir_name, i)
        img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
        # get a cropped map
        img_cropped = img_crop(img.copy())
        # save the cropped image
        output_filename = "%sDetectedArea%02d.jpg" % (outputdir_name, i)
        plt.imshow(img_cropped, cmap='gray')
        plt.savefig(output_filename)
        # separate each group of building numbers
        signage_contour_list = segment_signage(img_cropped)

        # # for debugging purpose
        # plt.figure()
        # cc = 1
        # for c in signage_contour_list:
        #     x, y, w, h = cv2.boundingRect(c)
        #     im_digits = img_cropped[y:y+h, x:x+w]
        #     plt.subplot(1,8, cc)
        #     plt.imshow(im_digits, cmap='gray')
        #     cc += 1
        # plt.show()

        # for each signage, open a file and report the classification results
        output_filename = "%sBuildingList%02d.txt" % (outputdir_name, i)
        f = open(output_filename, "w+")
        for j in range(len(signage_contour_list)):
            # segment the image to get digits region
            img_seg, contour_list = segment_digits(img_cropped.copy(), signage_contour_list[j])
            # classify the digits
            if len(contour_list) == 4:
                pred_labels = classify_digits(img_seg, contour_list[:3], knn_model_digit)
                pred_labels_arrow = classify_digits(img_seg, contour_list[3:], knn_model_arrow)
                # concatenating two groups of labels
                pred_labels.extend(pred_labels_arrow)
            else:
                pred_labels = classify_digits(img_seg, contour_list, knn_model_digit)
            # save the recognition output to a text file
            output_str = "Building "
            # printing
            # check whether four chars are localized, ideally 3 digits + 1 arrow should exist
            if (len(pred_labels) == 4):
                for i in range(3):
                    output_str += str(pred_labels[i])
                if pred_labels[3] == 0:
                    output_str += " to the left"
                else:
                    output_str += " to the right"
            else:
                for i in range(len(pred_labels)):
                    output_str += str(pred_labels[i])
            # add a newline to the current building signage string
            output_str += '\n'
            f.write(output_str)
        # close the file
        f.close()
