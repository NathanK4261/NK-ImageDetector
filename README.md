# nkImageDetector

A tool for detecting specific objects on a re-trained ssd-mobilenet model.

Built By: Nathan D. Keidel

# About

nkImageDetector was built using the "jetson-inference" library and uses the "detectNet" method. 
nkImageDetector provides a more "streamlined" process to building, training, and exporting your own ML models, while
also being more memory efficient

# System Requirements

In order to use "nkImageDetector", you MUST have
- A NVIDIA Jetson (Nano Developer Kit recomended for better compatibility)
  
- A computer running "Ubuntu x86_64" (For Jetsons that **ARE NOT** a Developer Kit)
  
- A microSD card flashed with JetPack and with python3.6 installed
  
- A USB Keyboard, Mouse, WiFi adapter, and Webcam (A USB hub is optional, but can help with I/O space

- A USB thumb drive above 5GB storage
  
- A 5V | 3A Usb-C power adapter
  
- A Google account with acces to "Google Drive"
  
- **(Optional)** VSCode downloaded on your personal computer

# Step 1: Setting up your Jetson

In order to run "nkImageDetector", you need to flash an OS onto your Jetson.
Look at the documentation from the official "jetson-inference" library to get
started with your Jetson 

- **(Note: DO NOT continue setting up your Jetson after you reached the end of
the document provided below, you will build the project in the next step)**

- [https://github.com/dusty-nv/jetson-inference/blob/master/docs/jetpack-setup-2.md](url)

# Step 2: Downloading Requirements

The next step is to download "jetson-inference". 
See the code provided below to build the project from source. 
(When building from source, Make sure to download PyTorch for python3.6 
since we are using python3.6)

- **(Note: DO NOT continue setting up your Jetson after you reached the end of
the code provided below, you will download your dataset in the next step)**

      sudo apt-get update
      sudo apt-get install git cmake libpython3-dev python3-numpy
      cd /home/nvidia
      git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
      cd jetson-inference
      mkdir build
      cd build
      cmake ../
      make -j$(nproc)
      sudo make install
      sudo ldconfig

# Step 3: Download dataset

The next step is to download a dataset.
I have made a Google Colab file that allows you to download images and train them on a remote
runtime, so you can save on memory and keep your Jetson from working too hard.

If you would like to download a dataset and train it **ON YOUR JETSON**, follow this document:

- [https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md](url)

If you would like to download a dataset on a **Google Runtime** follow these steps;

_For best results, follow these instructions and DO NOT go ahead._

- Open this link --> [https://colab.research.google.com/drive/1nnnpb2Dnh5Mkb87r-PjacwRT_p8Y0a1u?usp=sharing](url)

- Click **File -> Save a copy in Drive**

- Open the copied file in a new tab. (you can exit the other Colab tab)
  
- Run the first cell of code by pressing the "Play" button next to it. You will see a green checkmark when the cell has finished running:
<img width="1138" alt="Screenshot 2023-07-07 at 1 57 03 AM" src="https://github.com/NathanK4261/driver-distraction-helper/assets/78992074/197228bf-aa62-4b35-adf5-5ba31584cb87">


- Next, run the second cell by pressing the "Play" button again. Again, you will see a green checkmark indicating the cell has finished running:
<img width="1156" alt="Screenshot 2023-07-07 at 2 13 44 AM" src="https://github.com/NathanK4261/driver-distraction-helper/assets/78992074/8c8043fb-0107-4e50-8599-ea13de4da694">


- Go to Open Images --> [https://storage.googleapis.com/openimages/web/visualizer/index.html?](url)
- When you follow this link, search around the different classes of images and remember the name of the category/catergories you want to download.
- Now, return to your Colab file and look at the third cell. It should look like this:

      !cd pytorch-detection/ssd; python3 open_images_downloader.py --class-names="INSERT CLASS NAME[S]" --max-images=5000 --data=data/images
  **In the Colab file, change "INSERT CLASS NAME[S]" with however many categories of images you want.
  Make sure the catergories are inside the quotes and divided with commas ( , )**

  **DO NOT CHANGE THE "--data=data/images" ARGUMENT, IT IS NEEDED LATER ON!**

- Run the third cell like the other 2 cells. Wait until a green checkmark has appeared next to the third cell to continiue.

- Lets examine the fourth cell:

      !cd pytorch-detection/ssd; python3 train_ssd.py --model-dir=models/my-model --data=data/images --batch-size=2 --workers=2 --epochs=30

  You can change the "batch-size", "workers", and "epochs" values _(Batch size affects number of images processed per-cycle,
  workers affects how many parallel units are processing images, and epochs affects how many times you want the code to go through all
  downloaded images)_

  **DO NOT CHANGE "--model-dir=models/my-model" AND "--data=data/images", THEY ARE NEEDED LATER ON!**

- The fourth cell can take a long time, and can possibly randomly time-out on Google Colab.
  You can buy "Colab +" which allows for tasks to be ran 24/7 on the remote runtime.
  It also allows for much faster GPU's and 500 compute units!

- Once the fourth cell is finished, you should see a green checkmark on the side of the cell.


Thats it! You have trained your model. Now, you must export it to your Jetson.

**Keep the Colab tab open, as it is needed for "Step 5".**

# Step 4: Exporting model to .onnx

- In the colab file, run the fifth cell. When you see that green checkmark, run the sixth cell.
  This should download a .zip file of the model.
  
- Now, on the computer that is storing the .zip file, find the directory
  of the downloaded .zip file, open the terminal, and run this command:

      scp /path/to/zip_folder.zip <name of your Jetson>@<Your Jetson's IP adress>:/home/nvidia

- Now, on your Jetson. Open the terminal and run:

      cd /home/nvidia
      unzip models.zip
      rm models.zip
      git clone --recursive https://github.com/NathanK4261/NK-ImageDetector.git
      mv models/* /home/nvidia/NK-ImageDetector/my-project/detection/ssd/models
      rm -rf models/
      cd NK-ImageDetector/my-project/detection/ssd
      python3 onnx_export.py --model-dir=models/my-model

Thats it! You are done exporting your model as a .onnx file and you can now begin to run the code!

# Step 5: Run "helper.py"

Now that we have correctly set up the detection folder, open the terminal and
run these scripts:

      cd /home/nvidia/NK-ImageDetector/my-project/detection/ssd
      nano helper.py

You should now be in a text editor, navigate to this line in the code:


<img width="930" alt="Screenshot 2023-07-07 at 6 06 27 AM" src="https://github.com/NathanK4261/driver-distraction-helper/assets/78992074/2db7fcdd-44a6-4af5-a7e3-af6fa3f52032">


_"CLASS_IDS = [''] # Change this to the names of the labels in your model"_

- There should be a file called "labels.txt" in "/home/nvidia/NK-ImageDetector/my-project/detection/ssd/models/my-model/labels.txt".

- Take every word in that file **EXCEPT FOR THE WORK "BACKROUND"** and change the values in the python script to match the labels in the text file.

- For example, if this was my "labels.txt" file...
    BACKROUND
    Cat
    Dog

- I would edit the python code to look like:


<img width="925" alt="Screenshot 2023-07-07 at 6 12 35 AM" src="https://github.com/NathanK4261/driver-distraction-helper/assets/78992074/ab25fccd-453c-4e69-91f5-6ceaa67daf6a">

- Once you do that, pess "controll+x" --> "y" --> "ENTER" to save the updated .py file

- You can now run the hlper.py script!
  
      python3 helper.py models/my-model/ssd-mobilenet.onnx models/my-model/labels.txt 4

This should run the python script. Which means you are ready to detect objects with your retrained model!
