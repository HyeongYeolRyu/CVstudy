import numpy as np
import cv2
import os

def get_face_image(origin_img):
    recognizers = os.listdir("face_recognizer_trained_file/")
    recognizers = [os.path.join("face_recognizer_trained_file/", recognizer) for recognizer in recognizers]
    for recognizer in recognizers: # use all recognizers if face not found
        face_cascade = cv2.CascadeClassifier(recognizer)
        gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            face_img = faces[0] # get img from tuple
            face_cord = face_img
            x, y, w, h = face_img
            face_img = gray[y:y + h, x:x + w]
            break

    return (face_cord, face_img)


def get_X_and_y(input_type, folders, verbose=False):
    X = []
    face_cords = []
    y = []
    X_origin = []
    count = 0
    if verbose:
        print("=======================")
    for i, id in enumerate(folders):
        img_folder_names = os.listdir(id)
        imgs = [id + "/" + img for img in img_folder_names]
        for img in imgs:
            origin_img = cv2.imread(img)
            X_origin.append(origin_img)
            face_cord, face_img = get_face_image(origin_img)
            face_cords.append(face_cord)
            X.append(face_img)
            y.append("ID_" + str(i + 1))
            count += 1
            if verbose and count % 10 == 0:
                print(str(count) + " " + input_type + " data loaded.")
    if verbose and count % 10 != 0:
        print(str(count) + " " + input_type + " data loaded.")
    X = np.array(X)
    y = np.array(y)
    X_origin = np.array(X_origin)
    face_cords = np.array(face_cords)
    
    return X, y, X_origin, face_cords


def load_data(verbose=False):
    path_tr = "FaceDatabase/Train/"
    path_tr_dir = os.listdir(path_tr)
    tr_folders = [path_tr + str(folder) for folder in path_tr_dir]

    path_te = "FaceDatabase/Test/"
    path_te_dir = os.listdir(path_te)
    te_folders = [path_te + str(folder) for folder in path_te_dir]

    X_train, y_train, X_train_origin, X_train_face_cords = get_X_and_y("Train", tr_folders, verbose)
    X_test, y_test, X_test_origin, X_test_face_cords = get_X_and_y("Test", te_folders, verbose)
    
    if verbose:
        print("=======================")
        print("data loading completed.")

    return X_train, y_train, X_train_origin, X_train_face_cords, X_test, y_test, X_test_origin, X_test_face_cords


def normalize_image(data, size):
    # unpacking
    target_width, target_height = size 
    cur_width, cur_height = data.shape
    
    # When scaling image, consider interpolation.
    # upscaling : cv2.INTER_CUBIC(slow) or cv2.INTER_LINEAR
    # downscaling : cv2.INTER_AREA
    
    if cur_width < target_width and cur_height < target_height:
        # upscaling
        data = cv2.resize(data, size, interpolation = cv2.INTER_CUBIC)
    elif cur_width > target_width and cur_height > target_height:
        # downscaling
        data = cv2.resize(data, size, interpolation = cv2.INTER_AREA)
    else:
        # default : cv2.INTER_LINEAR
        data = cv2.resize(data, size, interpolation = cv2.INTER_LINEAR)
    
    return data


def check_image(image):
    cv2.imshow("sample", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def visualize_pred(faces, origin_image, y_pred):
    for i, img in enumerate(origin_image):
        x,y,w,h = faces[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, y_pred[i], (x-6, y-6), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()