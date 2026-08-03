from skimage.io import imread
#this gives the python script the ability to load and read the image files from the computer so that way i can work with them in the code 
#this converts it to a numpy array 
from skimage.filters import threshold_otsu
#this basically looks at the intesnity of the pxiels and finds the cutoff that minimizes the variance in teh dark and light pixel groups while maximixzing the contrast and ouyputs a single value 
#this seperates the plate text from the ackground glare, shadows, and paint 
import matplotlib.pyplot as plt
from skimage import measure
#this scans the grid of the true and false and groups the connected colors into numbered blobs so that way python knows where each blob begins and ends 
from skimage.measure import regionprops
#this calculates the geometry of each blob like the area and the coordinates 
import matplotlib.patches as patches
#once the regionprops give shapes so that way we know what t is going to be 
import numpy as np
from skimage.transform import resize
import os
#this gives python access to my operating sstem and this is needed when i need to check model files 
#this is like an organizer 
from sklearn.svm import SVC
#svc is something that sorts things like where i feed it pictures of letters and the program learns the patterns 
from sklearn.model_selection import cross_val_score
#this tells me how well my svc works 
import joblib
#this saves my trained bot onto the harddrive 
 
 
# ======================================================================
# CONSTANTS
# ======================================================================
 
letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
 
 
# ======================================================================
# TRAINING HELPERS
# ======================================================================
 
def read_training_data(training_directory):
    image_data = []  # this is to store the pixel arrays
    target_data = []  # this is for the actual answers
    for char in letters:
        for i in range(50):
            image_path = os.path.join(training_directory, char, f"{char}_{i}.jpg")
            # example:
            # each box has a letter written on it and then i loop it 10 times so that way there are 10 different ways to draw the letter
            image_details = imread(image_path, as_gray=True)
            binary_image = image_details < threshold_otsu(image_details)
            flat_bin_image = binary_image.reshape(-1)
            # this is to change it from a 2d grid to a 1d grid
            # -1 is so that way it figures out the exact size so that way i don't have to automatically calculate it
            image_data.append(flat_bin_image)
            # every time python finishes processing a picture and compressing it into 1d array, .append() will add it to the big pile of cards and this is kind of like the front of a flashcard
            # this saves what the image looks like
            target_data.append(char)
            # this saves what the image is actually called
            # this has the actual answer and like the real name and must be in the exact order
    return np.array(image_data), np.array(target_data)
    # we use a numPy array because finding and sorting takes more time using regular lists because python has to check them individually
 
 
def cross_validation(model, num_of_folds, train_data, train_label):
    # model is the type of model, #number of piles is the number of folds
    # train data is the features in pixels
    # train label is the answers
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_folds)
    # this tells me how accurate that my code/bot is
    print(f"Cross Validation Result for {num_of_folds} -fold:")
    print(str(accuracy_result * 100))
 
 
def train_and_save_model(current_directory):
    """Reads the training data, cross-validates, fits, and saves the SVC model."""
    training_dataset_dir = os.path.join(current_directory, 'train')
    # without this, the current dir is the main adress but train is the specific address (sub place) where the pictures are kept
    # join combines them into one full path
    # os.path.join is necessary because it is unviersal and we can't combine them with a + because different os uses diff syntaxes
    image_data, target_data = read_training_data(training_dataset_dir)
    # this saves the two returned arrays into two seperate variables
    # this gives the questions and answers to the data set to learn by making a training data set
    # this is the preparation
 
    svc_model = SVC(kernel="linear", probability=True)
    # linear tells it to try to draw straight lines and probability tells me the condifence scores with the predictions
    cross_validation(svc_model, 4, image_data, target_data)
    # don't need the . because cross_validation is a standalone function not a method that was built into svc_model
    svc_model.fit(image_data, target_data)
    # this is going to train the model with the input data
    # this is the actual learning
 
    save_directory = os.path.join(current_directory, 'models/svc/')
    # this is the folder path to save it and this is to store my trained model files
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        # this just creates a folder if one doesn't already exist
    joblib.dump(svc_model, save_directory + '/svc.pkl')
    # this just saves the fully trained file in a physical file and saving it to a pickle file, it freezes the brain
 
 
# ======================================================================
# PLATE LOCATION
# ======================================================================
 
def find_plate_like_objects(gray_car_image, ax1):
    """Binarizes the car image, labels connected regions, and collects
    the regions whose size looks like a license plate. Also draws every
    candidate region's bounding box onto ax1 for debugging."""
    threshold_value = threshold_otsu(gray_car_image)
    # threshold_otsu looks at the entire picture and calculates one single number that is the perfect brightness
    # this will automatically adjust the lighting that specific hoto making sure that the license plat letters pop out cleanly
    binary_car_image = gray_car_image > threshold_value
    # this asks if each pixel has a brightness greater than the threshold value and then it is a boolean that becomes black and white
    # True = WHITE
    # False = BLACK
 
    label_image = measure.label(binary_car_image)
    # this connects the white pixels into different groups and creates numbered groups and helps o measure individual objects
 
    plate_dimensions = (
        0.08 * label_image.shape[0],  # this is the minimum height
        0.2 * label_image.shape[0],  # this is the maximum height
        0.15 * label_image.shape[1],  # this is the minimum width
        0.4 * label_image.shape[1]  # this is the maximum width
    )
    # we use percentages of the shpae because i want it to be dynamic
    # the point of this is to prevent the code from identifying other objects as license plates
 
    min_height, max_height, min_width, max_width = plate_dimensions
    # this is tuple unpacking where each number in the tuple is assigned to a specific name
 
    plate_objects_coordinates = []
    # this is for storing the license plate's location that can help crop it
    plate_like_objects = []  # this is the actual cropped picure using the list of coordinates
 
    for region in regionprops(label_image):
        if region.area < 50:
            continue
        # region area counts the number of white pixels in yhe specific hrpup and then if it is less than 50 then it is probably not a license plate
        # true=1=white
        # scikit ignores the black pixels because it thinks that the black areas are empty
        # i was confused because i thought that the license plate numbers are black so why r we using white
        # this is because we're js looking for the plate here
        min_row, min_col, max_row, max_col = region.bbox
        # this is the bouding box and tells python to form a box around the group that we found
        region_height = max_row - min_row
        region_width = max_col - min_col
        # this helps decide if the shape is a license plate or something else
        # this is also needed for the red box
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width:
            plate_like_objects.append(binary_car_image[min_row: max_row, min_col: max_col])
            # saves the cropped image that is added to the list
            plate_objects_coordinates.append((min_row, min_col, max_row, max_col))
            # is added to the address
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                        edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        # we need this to see if our group was the right location for cpaturing the license
        # there will be multiply rectangles based on all the numbered groups
 
    return plate_like_objects, plate_objects_coordinates
 
 
# ======================================================================
# CHARACTER SEGMENTATION
# ======================================================================
 
def segment_characters(plate_like_objects):
    """Tries each plate candidate in order, and keeps the first one that
    actually segments into character-shaped blobs (instead of hardcoding
    plate_like_objects[2] like before)."""
    characters = []
    column_list = []
    license_plate = None
 
    best_plate = None
    best_characters = []
    best_column_list = []
 
    for candidate in plate_like_objects:
        test_plate = np.invert(candidate)
        # np.invert takes the photo and flips the color of the pixels where each black pixel becomes white and vice versa
        # this is because white is true and so this changes the background of the license plate to be black while the letters and numbers are white
 
        character_dimensions = (
            0.35 * test_plate.shape[0],  # min height
            0.60 * test_plate.shape[0],  # max height
            0.05 * test_plate.shape[1],  # min width
            0.15 * test_plate.shape[1]  # max width
        )
        min_height, max_height, min_width, max_width = character_dimensions
        char_label_image = measure.label(test_plate)
 
        test_characters = []
        test_column_list = []
        for region in regionprops(char_label_image):
            y0, x0, y1, x1 = region.bbox
            region_height = y1 - y0
            region_width = x1 - x0
            if min_height <= region_height and max_height >= region_height and min_width <= region_width and max_width >= region_width:
                roi = test_plate[y0:y1, x0:x1]
                resizedchar = resize(roi, (20, 20))
                resizedchar = resizedchar >0.5
                test_characters.append(resizedchar)
                test_column_list.append(x0)
 
        # keep this candidate only if it beats the best one found so far
        if len(test_characters) > len(best_characters):
            best_plate = test_plate
            best_characters = test_characters
            best_column_list = test_column_list
 
    return best_plate, best_characters, best_column_list
 # this used to just be "if test_characters: ... break" which meant the loop would
        # grab the very FIRST candidate that produced even 1 region matching the size window,
        # and then stop looking at the rest of plate_like_objects entirely
        # the problem with that is plate_like_objects isn't sorted by "most plate-like" -
        # it's just in whatever order regionprops happened to find them in the image
        # so a random blob (shadow, trim, glare) could accidentally have ONE little sliver
        # that fits inside character_dimensions, and the loop would lock onto that junk blob
        # and quit before it ever got to the real plate later in the list
        # switching "if test_characters" to "if len(test_characters) > len(best_characters)"
        # and removing the break means we check every single candidate instead of stopping early
        # then we keep whichever candidate produced the MOST character-shaped regions
        # a real plate has like 6-8 characters on it, so it should almost always win
        # against a stray blob that only managed to fake 1 or 2 by accident
 
 
def draw_character_boxes(license_plate, column_list, ax_plate):
    for region in regionprops(measure.label(license_plate)):
        y0, x0, y1, x1 = region.bbox
        if x0 in column_list:
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                             linewidth=2, fill=False)
            ax_plate.add_patch(rect_border)
 
 
# ======================================================================
# CLASSIFICATION
# ======================================================================
 
def classify_characters(characters, column_list, model):
    """Runs each segmented character through the trained model, then
    reorders the predicted letters left-to-right using column_list."""
    classification_results = []
    # this an empty ist to collect and hold of the predictions as the model goes through the list
    for char in characters:
        char = char.reshape(1, -1)
        # rows, columns
        result = model.predict(char)
        # predict was already part of the sci-kit learn library
        classification_results.append(result)
        # this becomes a list of individual arrays
 
    plate_string = ''
    for each in classification_results:
        plate_string += each[0]
        # this is like this because when we choose a single character, it is a numpyarray and the letter is stored at position 0 in the array
        # if it was just each -> ['K']
        # if it is each[0] -> 'K'
 
    column_list_copy = column_list[:]
    # this makes a duplicate copy of the list
    column_list.sort()  # -> rearranges the column list in order
    correct_plate_string = ''
    for each in column_list:
        correct_plate_string += plate_string[column_list_copy.index(each)]
        # each is the sorted position and uses the og position to get the right letter and puts it on the correctplate_string
 
    return correct_plate_string
 
 
# ======================================================================
# MAIN
# ======================================================================
 
def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # os.path.dirname is the adress and chopes off the file name and just keeps the folder where my script lives
    # realpath part gives me the full adress
 
    train_and_save_model(current_directory)
 
    car_image = imread("car.jpg", as_gray=True)
    # this changes it into a 2d array of grayscale intensity
    # shape is a property of the arrays that tells the dimensions of the grid
    # the true part of it confirms that the 2d grayscale matrix is a 3d color matrix
    # the purpose of this is to calculate the bouding boxes and pixel regions if i want to extract something later on
 
    gray_car_image = car_image * 255
    # this scales all pixel values in my image array convering them into the standard 8 bit image range
 
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #this sets up a window with two plotting areas next to each other
    #fig is the overall window frame and the subplots cretas a grid with 1 row and 2 columns
    #ax1 is on the left
    #ax2 is on the right
    fig, ax1 = plt.subplots(1)
    # fig is the overall window frame and then the ax1 is the drawing axis amd there is one overall canvas
 
    ax1.imshow(gray_car_image, cmap="gray")
    # imshow stands for image show and changes the 2d gtid pixels into visual pixels
    # cmap makes the picture gray
    # if i don't have this, then matplotlib will use colors that don't resemble grayscale
 
    plate_like_objects, plate_objects_coordinates = find_plate_like_objects(gray_car_image, ax1)
 
    license_plate, characters, column_list = segment_characters(plate_like_objects)
 
    if license_plate is None:
        print("No plate candidate produced character-shaped regions.")
        plt.show()
        raise SystemExit
 
    fig2, ax_plate = plt.subplots(1)
    # this creates a new display window and ax_plate is the drawing canvas
    ax_plate.imshow(license_plate, cmap="gray")
    draw_character_boxes(license_plate, column_list, ax_plate)
 
    model_dir = os.path.join(current_directory, 'models/svc/svc.pkl')
    model = joblib.load(model_dir)
 
    correct_plate_string = classify_characters(characters, column_list, model)
    print(correct_plate_string)
 
    plt.show()
 
 
if __name__ == "__main__":
    main()