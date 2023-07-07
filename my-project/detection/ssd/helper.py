# Import modules
import jetson_inference as ji
import jetson_utils as ju
import sys

# NK-Processing Class
# Methods for creating ways to easilly index labels in ssd-mobilenet models, as well as
# Display custom
def LabelMaker(labels_dir):
        
        labels_dir = labels_dir

        # Iterate through "labels.txt and create a list of labels"
        with open(labels_dir, 'r+') as file:
            labels = file.read()
            labels_list = labels.split()

        return labels_list

# Check if the user has given all required arguments
if len(sys.argv) < 3:
    print('''USAGE: [python] "helper.py"
<model-dir> - Path to model from current directory. (For example: models/model/onxx_model.onnx)
<label-dir> - Path to the text file of labels from current directory. (For example: models/model/labels.txt)
<threshold> - Integer representing minimum detection threshold.
''')
    sys.exit(0)

# This variable is to detect if our user has pressed the "Q" key twice so we can quit the program
Q_key_pressed_counter = 0
display = None # Keep as "None" object so we can check if a display is connected or not

# Load our camera from a USB
try:
    cam = ju.videoSource('/dev/video0')

    # Check if user requested the disply to be active
    try:
        display = ju.videoOutput('display://0')
        print('''
              
              CONNECTED TO DISPLAY
              
              ''')
    except:
        print('''
              
              NO DISPLAY DETECTED
              
              ''')
        False

except:
    print('ERROR: There was an error loading either the camera or the display')

# Load our net
net = ji.detectNet(argv=['--model=' + sys.argv[1], 
                         '--labels=' + sys.argv[2], 
                         '--input-blob=input_0', 
                         '--output-cvg=scores', 
                         '--output-bbox=boxes'], 
                         threshold = int(sys.argv[3]))

print('''


    STARTING VIDEO FEED


''')
      

# Load class ID's in the order they appear in at "labels.txt"
CLASS_IDS  =  LabelMaker(str(sys.argv[2]))# Change this to the names of the labels in your model
print(CLASS_IDS)

# Read an image and detect possible road obstacles
while True:
    img = cam.Capture()

    # Check if we have taken a photo
    if img is None:
        break

    # Detect objects in the frame
    detections = net.Detect(img)

    # Display the frame and set status if we have a display connected
    if display != None:
        display.Render(img)
        display.SetStatus("Object detection | {:.0f} FPS".format(net.GetNetworkFPS()))
    
    # Iterate through the detected images and print what classes have been identified
    for detected_class in detections:
        # Take the class ID and subtract 1 from it since the frist class ID in "labels.txt" is BACKROUND
        classID = int(detected_class.ClassID)

        # Pring what class we have detected
        print('Detected:', CLASS_IDS[classID], 'SLOW DOWN')
