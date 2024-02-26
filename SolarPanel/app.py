import uvicorn
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import pickle
import numpy as np
import os
import requests
import subprocess
import shutil
import easyocr
from yolov5.detect import counter

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
return_value=0    
UPLOAD_FOLDER = "images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_text",response_class=HTMLResponse)
async def upload_text(request: Request,userInput: str = Form(...)):
    
    def get_lat_lng(location, api_key):
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": location,  
            "key": api_key
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if data['status'] == 'OK':
            results = data['results'][0]
            location = results['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            return latitude, longitude
        else:
            print("Error:", data['status'])
            return None, None

    # Ask the user to enter the location
    location = userInput
              
    # Replace 'YOUR_API_KEY' with your actual Google Maps API key
    api_key = "AIzaSyB8zWPtv1G6B05tim27903BAeUQXjGS9dc"
     
    # Get the latitude and longitude
    latitude, longitude = get_lat_lng(location, api_key)

    if latitude is not None and longitude is not None:
        print("Latitude:", latitude)
        print("Longitude:", longitude)
    else:
        print("Failed to retrieve latitude and longitude.")

    def fetch_static_map(latitude, longitude, api_key):
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{latitude},{longitude}",
            "zoom": 19,  # Adjust the zoom level as needed
            "size": "400x400",
            "maptype": "satellite",
            "key": api_key
        }
        response = requests.get(base_url, params=params)
        return response.content

   
    api_key = "AIzaSyB8zWPtv1G6B05tim27903BAeUQXjGS9dc"

    static_map_image = fetch_static_map(latitude, longitude, api_key)

    with open("static/static_map_image1.jpg", "wb") as f:
           f.write(static_map_image)
    
    rpred=process_image("static/static_map_image1.jpg")
    
    image_path = 'static/output.jpg'
    img = cv2.imread(image_path)

    # # instance text detector
    # reader = easyocr.Reader(['en'], gpu=False)

    # # detect text on image
    # text_ = reader.readtext(img)

    # # count occurrences of 'solar_panel'
    # word_to_count = 'solar_panel'
    # predictions = sum(1 for _, text, _ in text_ if word_to_count in text.lower())

      
         
    context = {
        "request": request,
        "predictions" : rpred
    }
    return templates.TemplateResponse("result.html", context)



    
     




# @app.post("/upload_image", response_class=HTMLResponse)
# async def upload_image( request: Request,image_file: UploadFile = File(...)):
#     image_path = f"{UPLOAD_FOLDER}/{image_file.filename}"
#     save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
#     with open(save_path, "wb") as image:
#         content = await image_file.read()
#         image.write(content)
    
#     predictions = process_image(image_path)
    
#     context = {
#         "request": request,
#         "predictions": predictions,
#         "uploaded_image": image_path    
#     }

#     return templates.TemplateResponse("index.html", context)

def process_image(file_path):
                            # with open('model.pkl', 'rb') as file:
                            #     model = pickle.load(file)

                            # input_image_path = file_path
                            # input_image = cv2.imread(input_image_path)
                            # input_image_resized = cv2.resize(input_image, (101, 101))  # Resize to match the model input size
                            # input_image_rescaled = input_image_resized / 255.0  # Scale pixel values between 0 and 1
                            # input_image_reshaped = np.expand_dims(input_image_rescaled, axis=0)  # Add batch dimension

                            # predictions = model.predict(input_image_reshaped).reshape((-1, ))
                            # binary_predictions = (predictions > 0.5).astype(int)

                            # print(binary_predictions)
                            
                            # return binary_predictions
    source_image = file_path
    weights_path = 'yolov5/best.pt'
    det_path='yolov5/detect.py'
    pred=run_detection(source_image, weights_path,det_path)
    return int(pred)
    


def run_detection(source_image, weights_path,det_path):
    
    command = ["python", det_path, "--source", source_image, "--weights", weights_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return_value = stdout.decode().strip()
    print(return_value) 
    
    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode('utf-8')}")
    else:
        print(f"Detection successful:\n{stdout.decode('utf-8')}")
    
    
    exp_folders = [folder for folder in os.listdir("yolov5/runs/detect") if folder.startswith("exp") and folder[3:].isdigit()]

    # If there are no exp folders, exit or handle the case accordingly
    if not exp_folders:
        print("No exp folders found.")
        exit()
      
    # Sort the exp folders by their numerical value
    sorted_exp_folders = sorted(exp_folders, key=lambda x: int(x[3:]))

    # Select the exp folder with the highest number
    latest_exp_folder = sorted_exp_folders[-1]

    # Get the full path of the latest exp folder
    latest_exp_folder = os.path.join("yolov5/runs/detect", latest_exp_folder)

    os.rename(f"{latest_exp_folder}/static_map_image1.jpg",f"{latest_exp_folder}/output.jpg")
    
    destination_folder = "static"
    output_file = "output.jpg"

    # Check if the file already exists in the destination folder
    destination_path = os.path.join(destination_folder, output_file)
    if os.path.exists(destination_path):
        os.remove(destination_path)
        print(f"Existing file '{output_file}' removed from '{destination_folder}'.")

    # Move the new file to the destination folder
    shutil.move(f"{latest_exp_folder}/{output_file}", destination_folder)

    return return_value
    
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
