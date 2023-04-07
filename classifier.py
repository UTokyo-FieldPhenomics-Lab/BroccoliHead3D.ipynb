import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from skimage import io, color


def expand_colorspace(img):
    # img is the ndarray from io.imread()
    (h, w, dimen) = img.shape
  
    if dimen == 4: # exist alpha layer (RGBA)
        img = img[:,:, 0:3]
        dimen = 3
    
    rgb = img.reshape(h * w, dimen)
    lab = color.rgb2lab(img).reshape(h * w, dimen)
    hsv = color.rgb2hsv(img).reshape(h * w, dimen)
    xyz = color.rgb2xyz(img).reshape(h * w, dimen)
  
    exp_img = np.concatenate((rgb, lab, hsv, xyz), axis=1)
  
    return exp_img
   
def training_data_generate(img_list, kind_list):
    # img_list = [io.imread(), io.imread(), io.imread()], element is ndarray
    # king_list = [0, 1, 2]
    train_data = np.empty((0, 12))
    train_kind = np.empty((0,))
    for img, kind in zip(img_list, kind_list):
        (h, w, dimen) = img.shape
        exp_img = expand_colorspace(img)
    
        if dimen == 4:
            img_reshape = img.reshape(h * w, dimen)
            train_index = np.nonzero(img_reshape[:, 3])  # the index of alpha layer not zero pixels
            exp_img = exp_img[train_index]  # pick the pixels that not background
    
        print(f"|--- Convert kind [{kind}] to training data, converted shape is {str(exp_img.shape)}")
        kinds = np.array([kind] * exp_img.shape[0]) 

        # merge each train_data and kind
        train_data = np.vstack((train_data, exp_img))
        train_kind = np.hstack((train_kind, kinds))

    return train_data, train_kind  
  
def train_model(train_data, train_kind, classifier="CART"):
    # classifier = ["CART", "SVM", "RF"]
    t0 = time.time()
    if classifier == "CART":
        clf = DecisionTreeClassifier(max_depth=20)
    elif classifier == "SVM":
        clf = LinearSVC()
    elif classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "GDBT":
        clf = GradientBoostingClassifier()
    
    clf = clf.fit(train_data, train_kind)
    t1 = time.time()
  
    print(f'|-- Training model time cost={int(t1-t0)}s')
    return clf
  
def predict_model(clf_img, model):
    (h, w, dimen) = clf_img.shape
    exp_clf_img = expand_colorspace(clf_img)
    pred_result = model.predict(exp_clf_img).reshape(h, w)
    return pred_result
  
def apply_model(clf_img, model):
    t0 = time.time()
    (h, w, dimen) = clf_img.shape
    if h * w >= 1500 * 1500:  # slice picture to save RAM
        slice_num = np.ceil(w * h / (1500 * 1500)).astype(np.int32)
        break_points = np.linspace(0, h, num=slice_num).astype(np.int32)
        line_st = break_points[:-1]
        line_ed = break_points[1:]
    else:
        line_st = [0]
        line_ed = [h]
  
    pred_result = np.empty((0, w))
    i = 0
    for st, ed in zip(line_st, line_ed):
        clf_result = predict_model(clf_img[st:ed, :], model)
        pred_result = np.vstack((pred_result, clf_result))
    
        percent = round((i+1) / len(line_st) * 100, 2)
        print('\r|' + '='*round(percent/2) + '>'+ ' '*round(50-percent/2) + '|' + str(percent) + '%', end="")
        i += 1
    t1 = time.time()
    print(f'| Cost={int(t1-t0)}s')
    return pred_result
  
def save_result(pred_result, img_save_name, color_list):
    color_list = np.array(color_list)/255
    cmap_cf = ListedColormap(color_list, name='colorfriendly')
    plt.imsave(img_save_name, pred_result, cmap=cmap_cf)
