import pyttsx3
import speech_recognition as sr
import datetime
import pyautogui
import time
import os
import googlemaps
from datetime import datetime
import requests
import cv2
import face_recognition
import numpy as np
import wolframalpha


OPENCAGE_API_KEY = 'b644863564e44e1c82c351cc84a72ae3'

# Replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual API key
OWM_API_KEY = 'cfca3b55c38affc4a2cbcbec5c719fe5'

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


engine.setProperty('rate', 150)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")
    elif hour>=12 and hour<18:
        speak("Good Afternoon!") 
        
    else:
        speak("Good Evening!")
    print('My Name Is Vision')
    speak('My Name Is Vision')
    print('I was created by Sakshi')
    speak('I was created by Sakshi')
    print('I can Do Everything that my creator programmed me to do')
    speak('I can Do Everything that my creator programmed me to do')
    print('Ready To Comply. What can I do for you ?')
    speak("Ready To Comply. What can I do for you ?")

def takeCommand():
    r =sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
        except Exception as e:
                print("Say that again please...")
                return "None"
        return query

def askAnyQuestion(command):
   client=wolframalpha.Client('5R6R75-2A9GXEPJ8V')
   n = input('How Many Question Do You Want Answer Of: ')
   n=int(n)
   while(n>0):
    question = str(input('What Is Your Question? ')) 
    res = client.query(question)
    if res['@success']=='false':
        print('No results')
    else:
        result = next(res.results, None)
        print(result.text if result else 'No results')
    n-=1

def get_coordinates(location):
    """Get the latitude and longitude of a location using OpenCage API."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    return None, None



def get_directions(start, end):
    """Get directions from start to end using OpenRouteService API."""
    start_lat, start_lng = get_coordinates(start)
    end_lat, end_lng = get_coordinates(end)
    
    if start_lat is None or end_lat is None:
        print("I couldn't find one of the locations. Please try again.")
        return
    
    # OpenRouteService Directions API URL
    url = f"https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        'Authorization': OPENCAGE_API_KEY,
        'Content-Type': 'application/json'
    }
    body = {
        "coordinates": [[start_lng, start_lat], [end_lng, end_lat]],
        "instructions": True
    }
    
    response = requests.post(url, json=body, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if 'routes' in data and data['routes']:
            steps = data['routes'][0]['segments'][0]['steps']
            directions = [step['instruction'] for step in steps]
            
            # Simulate speaking each direction step-by-step
            for direction in directions:
                speak(direction)
        else:
            print("No routes found.")
    else:
        print("Error retrieving directions.")





def get_weather(city):
    """Fetch the weather information for a given city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"The weather in {city} is currently {weather} with a temperature of {temperature}°C."
    else:
        return "Sorry, I couldn't fetch the weather information."
    


# Load YOLO
# Define paths
config_path = r"C:\Users\Administrator\Desktop\Vision\yolov3.cfg"
weights_path = r"C:\Users\Administrator\Desktop\Vision\yolov3.weights"
coco_names_path = r"C:\Users\Administrator\Desktop\Vision\coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(frame):
    """Detect objects in a given frame using the YOLO model."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objects.append((label, x, y, w, h))
    return objects

def handle_command(command):
    if 'detect objects' in command:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            objects = detect_objects(frame)
            for obj in objects:
                label, x, y, w, h = obj
                # Draw a rectangle around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put the label of the object on the screen
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Speak the label of the detected object
                print(f"I see a {label}")
                speak(f"I see a {label}")
                
                
            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


known_face_encodings = []
known_face_names = []

def load_known_faces():
    # Load and encode known faces (you'll need to have images of known faces)
    image = face_recognition.load_image_file("known_face.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("Person Name")

def recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names

if __name__ == "__main__":
    wishMe()
    while True:
        command = takeCommand().lower()
        if 'vision' in command:
            print('yes master')
            speak('yes master')
            command = takeCommand().lower()
            if "Questions" in command:
                askAnyQuestion(command)

            elif "weather" in command:
                speak("Which city's weather would you like to know?")
                city = takeCommand()
                weather_info = get_weather(city)
                print(weather_info)
                speak(weather_info)

            elif "directions" in command:
                 speak("Where do you want to start?")
                 start = takeCommand()
                 speak("Where do you want to go?")
                 end = takeCommand()
                 get_directions("1600 Amphitheatre Parkway, Mountain View, CA", "1 Infinite Loop, Cupertino, CA")

            elif "time" in command:
                now = datetime.now().strftime("%H:%M")
                print(f"The current time is {now}") 
                speak(f"The current time is {now}")
            
            elif 'who is' in command or 'what is' in command:
                client = wolframalpha.Client('5R6R75-2A9GXEPJ8V')
                res = client.query(command)
                try:
                    answer = next(res.results).text
                    print(answer)
                    speak(answer)
                except StopIteration:
                    speak("I could not find an answer to your question.")
                    
                    """Handle commands to perform actions based on the command."""
            
            elif 'detect objects' in command or 'recognize faces' in command:
                handle_command(command)

            elif 'recognize faces' in command:
                load_known_faces()
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    face_names = recognize_faces(frame)
                    for name in face_names:
                        speak(f"I see {name}")
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                cap.release()
                cv2.destroyAllWindows()
                break
            elif "stop" in command or "exit" in command or "quit" in command:
                speak("Goodbye!")
                break
        else:
            speak("I am not sure how to help with that.")
                