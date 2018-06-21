from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import cv2
import numpy as np


def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image

def extract_features(imgs,  orient=9, pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_f,_= get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                 feature_vec=True)
            hog_features.append(hog_f)
        
        hog_features = np.ravel(hog_features)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

def find_cars(img, ystart, ystop, scale, svc, orient, 
              pix_per_cell, cell_per_block, show_all_rectangles=False):
    
    # array of rectangles where cars were detected
    rectangles = []
    
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]

    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1]//scale), int(imshape[0]//scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1,_ = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
    hog2,_ = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3,_ = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            test_prediction = svc.predict([hog_features])
            
            if test_prediction[0] == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return rectangles
    

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image and final rectangles
    return img, rects



def process_frame(img):

    rectangles = []
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    search_areas_list =[ [400 ,464 ,1.0], [416 ,480 ,1.0], [400 ,496 ,1.5], [432 ,528 ,1.5], [400 ,528 ,2.0], [432 ,560 ,2.0], [400 ,596 ,3.5], [464 ,660 ,3.5] ]
    for search_area in search_areas_list:
        rectangles.append(find_cars(img,*search_area , svc, orient, pix_per_cell, cell_per_block))

    rectangles = [item for sublist in rectangles for item in sublist]

    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_bboxes(np.copy(img), labels)
    return draw_im
    
    
    
    
    # Define a class to store data from video
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]
    
    def process_frame(img):
        rectangles = []
        orient = 11
        pix_per_cell = 16
        cell_per_block = 2
        search_areas_list =[ [400 ,464 ,1.0], [416 ,480 ,1.0], [400 ,496 ,1.5], [432 ,528 ,1.5], [400 ,528 ,2.0], [432 ,560 ,2.0], [400 ,596 ,3.5], [464 ,660 ,3.5] ]
        for search_area in search_areas_list:
            rectangles.append(find_cars(img,*search_area , svc, orient, pix_per_cell, cell_per_block))
        rectangles = [item for sublist in rectangles for item in sublist]

        if len(rectangles) > 0:
            self.add_rects(rectangles)
        heatmap_img = np.zeros_like(img[:,:,0])
        for rect_set in self.prev_rects:
            heatmap_img = add_heat(heatmap_img, rect_set)
        heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)
     
        labels = label(heatmap_img)
        draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
        return draw_img
