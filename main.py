from tkinter import *
from PIL import Image, ImageTk, ImageOps
import cv2
import time
import pyttsx3
from tkinter import messagebox,filedialog
import numpy as np
import matplotlib.pyplot as mlt
import tensorflow as tf
from random import randint
from datetime import datetime


model= tf.saved_model.load('ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model')
labels = ['background', 'human', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
          'bench', 'bird', 'cat', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
          'zebra', 'giraffe', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'tie', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'kite',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'apple', 'pizza', 'donut', 'cake',
          'carrot', 'couch', 'pizza', 'bed', 'dining table', 'chair', 'tv', 'laptop', 'mouse',
          'remote', 'keyboard', 'cell phone', 'microwave', 'toilet', 'toaster', 'tv', 'laptop',
          'book', 'remote', 'keyboard', 'teddy bear', 'hair drier', 'toothbrush']

# # Adding video detection to model
# video_path = 'path_to_your_video_file.mp4'
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to a format suitable for your model
#     input_tensor = tf.convert_to_tensor([frame])
#     detections = model(input_tensor)

#     # Process detections
#     for detection in detections['detection_boxes']:
#         ymin, xmin, ymax, xmax = detection[0].numpy()
#         (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
#                                       ymin * frame.shape[0], ymax * frame.shape[0])

#         # Draw bounding box
#         cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

#         # Draw label
#         label = 'Object Name'  # Replace with actual label from detection results
#         cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     # Write the frame to output video if needed
#     # out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


engine= pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
female_voice = None
for voice in voices:
    print(f"ID: {voice.id}, Name: {voice.name}, Gender: {'Female' if 'female' in voice.name.lower() else 'Male'}")
    if "female" in voice.name.lower():
        female_voice = voice.id
        break

if female_voice:
    engine.setProperty('voice', female_voice)
else:
    print("No female voice found. Using default voice.")
root = Tk()
root.geometry("1150x800")
root.title("Object Detection System")

i_img= Image.open("icon.jpg")
i_img = ImageTk.PhotoImage(i_img)
root.iconphoto(False, i_img)

def start_system():
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    engine.say(f"Hello, Welcome to the object recognition system. It's {current_time}")
    engine.runAndWait()

    original_rate = engine.getProperty('rate')
    slower_rate = original_rate - 20  
    engine.setProperty('rate', slower_rate)

    engine.say('Select "Capture Image" to capture the image of an object.')
    engine.runAndWait()
    engine.say('Select "Detect Object" to detect the object in an image.')
    engine.runAndWait()
    engine.say('And select "Exit" to exit the software.')
    engine.runAndWait()

def capture_img():
    engine.say("Taking picture")
    engine.runAndWait()
    
    vid = cv2.VideoCapture(0)
    start_time = time.time()
    image = None
    while True:
        ret, frame = vid.read()
        cv2.imshow('camera', frame)
        if time.time() - start_time >= 3:
            image = frame
            break
        if cv2.waitKey(1) & 0xff == ord('a'):
            image = frame
            break

    vid.release()
    cv2.destroyAllWindows()

    if image is not None:
        cv2.imwrite("new_image.png", image)
        engine.say("Image capture successfully")
        engine.runAndWait()
        messagebox.showinfo("Success", "Image Capture Successfully")
    else:
        messagebox.showerror("Error", "Failed to capture image")

def select_i():
    # Open the file dialog to browse for an image file
    engine.say("Choose an object to detect")
    engine.runAndWait()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp;*.gif;*.bmp")])
    
    if file_path:
        # Load the image file using PIL
        img = Image.open(file_path)
        img_arr= np.array(img)
        engine.say('detecting the object')
        engine.runAndWait()
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_arr,0))
        image_detection= model(input_tensor)
        boxes= image_detection['detection_boxes'].numpy()
        classes= image_detection['detection_classes'].numpy().astype(int)
        scores= image_detection['detection_scores'].numpy()
        
        for i in range(classes.shape[1]):
            class_id= int(classes[0,i])
            score= scores[0,i]
            if np.any(score > 0.5) and class_id < len(labels):
                h,w,_= img_arr.shape
                ymin,xmin,ymax,xmax= boxes[0,i]
                xmin= int(xmin*w)
                ymin= int(ymin*h)
                xmax= int(xmax*w)
                ymax = int(ymax*h)
                class_name= labels[class_id]

                random_color= (randint(0,256),randint(0,256),randint(0,256))
                cv2.rectangle(img_arr, (xmin,ymin), (xmax,ymax), random_color, 2)
                label= f"class:{class_name}, score:{score:.2f}"
                cv2.putText(img_arr,label,(xmin,ymin-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,random_color,2)
        
        if np.any(scores > 0.5):
            detected_classes = [labels[int(classes[0, i])] for i in range(classes.shape[1]) if scores[0, i] > 0.5]
            detected_objects = ", ".join(detected_classes)

            if len(detected_classes) <= 1:
                engine.say('object detected successfully')
                engine.runAndWait()
                engine.say(f'and Object is: {detected_objects}')
                engine.runAndWait()
                messagebox.showinfo('Success', f'Object detected: {detected_objects}')
            else:
                engine.say('objects detected successfully')
                engine.runAndWait()
                engine.say(f'and Object are: {detected_objects}')
                engine.runAndWait()
                messagebox.showinfo('Success', f'Objects detected: {detected_objects}')

            print(class_name)
            print(score)
        else:
            engine.say('Failed to detect objects')
            engine.runAndWait()
            messagebox.showerror('Error', 'Failed to detect objects')
        
        mlt.imshow(img_arr)
        mlt.axis('off')
        mlt.show()
        
def exit():
    engine.say('Okay i am going offline...     have a good day')
    engine.runAndWait()
    root.destroy()

img1= Image.open("hello.png")
img1= img1.resize((385,180), Image.Resampling.LANCZOS)
img1 = ImageTk.PhotoImage(img1)

f_lbl= Label(root, image= img1)
f_lbl.place(x=0, y=0)

img2= Image.open("img2.webp")
img2= img2.resize((500,180), Image.Resampling.LANCZOS)
img2= ImageTk.PhotoImage(img2)

s_lbl= Label(root, image=img2)
s_lbl.pack()

img3= Image.open("img3.png")
img3= img3.resize((385,180), Image.Resampling.LANCZOS)
img3= ImageTk.PhotoImage(img3)

t_lbl= Label(root, image=img3)
t_lbl.place(x= 890,y=0)

title= Label(root, text="OBJECT DETECTION SYSTEM", font='Arial 25 bold', bg='white', fg='blue')
title.pack(padx=320)

#background img
bg_img= Image.open("background image.webp")
bg_img= bg_img.resize((1275,450), Image.Resampling.LANCZOS)
bg_img= ImageTk.PhotoImage(bg_img)

t_lbl= Label(root, image=bg_img)
t_lbl.place(x=0,y=240)

img4= Image.open("img1.jpeg")
img4= img4.resize((140,110), Image.Resampling.LANCZOS)
img4= ImageTk.PhotoImage(img4)

b1= Button(root, image=img4)
b1.place(x=30 ,y=280)

b1_1= Button(root, text='Start', cursor='hand2', font='Arial 15 bold', bg= 'white', fg="gray", command= start_system)
b1_1.place(x=34 ,y=390, width=139, height=35)

img5= Image.open("Camera.png")
img5= img5.resize((150,110), Image.Resampling.LANCZOS)
img5= ImageTk.PhotoImage(img5)

b1= Button(root, image=img5)
b1.place(x=405 ,y=280)

b1_1= Button(root, text='Capture Image', cursor='hand2', font='Arial 15 bold', bg= 'white', fg="gray", command= capture_img)
b1_1.place(x=410 ,y=390, width=148, height=35)

img6= Image.open("detect.webp")
img6= img6.resize((150,110), Image.Resampling.LANCZOS)
img6= ImageTk.PhotoImage(img6)

b1= Button(root, image=img6)
b1.place(x=710 ,y=280)

b1_1= Button(root, text='Detect Object', cursor='hand2', font='Arial 15 bold', bg= 'white', fg="gray",command=select_i)
b1_1.place(x=715 ,y=390, width=148, height=35)
lbl1= Label(root)

img7= Image.open("Exit.webp")
img7= img7.resize((150,110), Image.Resampling.LANCZOS)
img7= ImageTk.PhotoImage(img7)

b1= Button(root, image=img7)
b1.place(x=1065 ,y=280)

b1_1= Button(root, text='Exit', cursor='hand2', font='Arial 15 bold', bg= 'white', fg="gray", command= exit)
b1_1.place(x=1070 ,y=390, width=148, height=35)

mainloop()