from django.shortcuts import render ,redirect
from django.contrib import messages
# Create your views here.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from .models import *
from django.http import HttpResponse
from sklearn.neighbors import KNeighborsClassifier
import cgi
import cgitb;cgitb.enable()
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm


def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')




def register(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        c_password = request.POST['c_password']
        if password == c_password:
            if user.objects.filter(email=email).exists():
                return render(request, 'register.html', {'message': 'User with this email already exists'})
            new_user = user(name=name, email=email, password=password)
            new_user.save()

            return render(request, 'login.html', {'message': 'Successfully Registerd!'})
        return render(request, 'register.html', {'message': 'Password and Conform Password does not match!'})
    return render(request, 'register.html')


def login(request):
    if request.method == "POST":
        email = request.POST['email']
        password1 = request.POST['password']
        
        try:
            user_obj = user.objects.get(email=email)
            print(3333, user_obj)
        except user.DoesNotExist:
            return render(request, 'login.html', {'message': 'nvalid Username or Password!'})
        
        password2 = user_obj.password
        if password1 == password2:
            return redirect('home')
        else:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})
    return render(request, 'login.html')

def home(request):
    return render(request, 'home.html')


#soil prediction
def soilprediction(request):
    global x_train, x_test, y_train, y_test, x, y
    if request.method == 'POST':
        df = pd.read_csv(r'datasets\data.csv')
        le = LabelEncoder()
        df['Output'] = le.fit_transform(df['Output'])
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=72)

        pH = float(request.POST['PH'])
        EC = float(request.POST['EC'])
        OC = float(request.POST['OC'])
        OM = float(request.POST['OM'])
        N = float(request.POST['N'])
        P = float(request.POST['P'])
        K = float(request.POST['K'])
        Zn = float(request.POST['Zn'])
        Fe = float(request.POST['Fe'])
        Cu = float(request.POST['Cu'])
        Mn = float(request.POST['Mn'])
        Sand = float(request.POST['Sand'])
        Silt = float(request.POST['Silt'])
        Clay = float(request.POST['Clay'])
        CaCO3 = float(request.POST['CaCO3'])
        CEC = float(request.POST['CEC'])
        p = 0.5
        xgp = p>=np.random.rand()
        PRED = [[pH, EC, OC, OM, N, P, K, Zn, Fe,Cu, Mn, Sand, Silt, Clay, CaCO3, CEC]]

        knn = RandomForestClassifier()
        knn.fit(x_train, y_train)
        xgp = np.array(knn.predict(PRED))
        

        if xgp == 0:

            msg = ' This prediction result is : Non Fertile'
        elif xgp == 1:
            msg = ' This prediction result is : Fertile'
        return render(request, 'soilprediction.html', {'msg': msg})
    return render(request, 'soilprediction.html')


#crop predictiopn
def croppredictiopn(request):
    df=pd.read_csv("datasets/Crop_recommendation.csv")
    global x_trains,y_trains
    le=LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    x = df.drop(['label'], axis=1)
    y = df['label']

    x_trains, x_tests, y_trains, y_tests = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)
   
    if request.method == "POST":
        f1 = float(request.POST['N'])
        print(f1)
        f2 = float(request.POST['P'])
        print(f2)
        f3 = float(request.POST['K'])
        print(f3)
        f4 = float(request.POST['temperature'])
        print(f4)
        f5 = float(request.POST['humidity'])
        print(f5)
        f6 = float(request.POST['ph'])
        print(f6)
        f7 = float(request.POST['rainfall'])
        print(f7)

        li = [[f1,f2,f3,f4,f5,f6,f7]]
        print(li)
        
        logistic = RandomForestClassifier()
        logistic.fit(x_trains, y_trains)
        
        result =logistic.predict(li)
        result=result[0]
        if result==0:
            msg = 'The Recommended Crop is predicted as Apple'
        elif result ==1:
            msg= 'The Recommended Crop is predicted as Banana'
        elif result ==2:
            msg= 'The Recommended Crop is predicted as Blackgram'
        elif result ==3:
            msg= 'The Recommended Crop is predicted as Chickpea'
        elif result ==4:
            msg= 'The Recommended Crop is predicted as Coconut'
        elif result ==5:
            msg= 'The Recommended Crop is predicted as Coffee'
        elif result ==6:
            msg= 'The Recommended Crop is predicted as Cotton'
        elif result ==7:
            msg= 'The Recommended Crop is predicted as Grapes'
        elif result ==8:
            msg= 'The Recommended Crop is predicted as Jute'
        elif result ==9:
            msg= 'The Recommended Crop is predicted as Kidneybeans'
        elif result ==10:
            msg= 'The Recommended Crop is predicted as Lentil'
        elif result ==11:
            msg= 'The Recommended Crop is predicted as Maize'
        elif result ==12:
            msg= 'The Recommended Crop is predicted as Mango'
        elif result ==13:
            msg= 'The Recommended Crop is predicted as Mothbeans'
        elif result ==14:
            msg= 'The Recommended Crop is predicted as Moongbeans'
        elif result ==15:
            msg= 'The Recommended Crop is predicted as Muskmelon'
        elif result ==16:
            msg= 'The Recommended Crop is predicted as Orange'
        elif result ==17:
            msg= 'The Recommended Crop is predicted as Papaya'
        elif result ==18:
            msg= 'The Recommended Crop is predicted as Pigeonpeas'
        elif result ==19:
            msg= 'The Recommended Crop is predicted as Pomegranate'
        elif result ==20:
            msg= 'The Recommended Crop is predicted as Rice'
        elif result ==21:
            msg= 'The Recommended Crop is predicted as Watermelon'
        
        return render(request,'croppredictiopn.html',{'msg':msg})
   
    return render(request, 'croppredictiopn.html')


Plants = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Potato___Early_blight',
            'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

def plantprediction(request):
    if request.method == 'POST':
        print("hdgkj")
        m = int(request.POST["alg"])
        acc = pd.read_csv("app\Accuracy.csv")
        file = request.FILES['file']
        full_path = os.path.join('app\static\img', file.name)
        # Ensure the directories leading up to the file exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Save the file
        with open(full_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        path = f'app/static/img/{file.name}'
        print(path)
        print('hjhdfakjhdaf-=-=-=-=-')

    

        if m == 2:
            print("bv2")
            new_model = load_model('app/models/ANN.h5')
            test_image = image.load_img(path, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image /=255
            a = acc.iloc[m - 1, 1]

        elif m == 3:
            print("bv2")
            new_model = load_model('app/models/CNN.h5')
            test_image = image.load_img(path, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image /=255
            a = acc.iloc[m - 1, 1]

        else:
            print("bv3")
            new_model = load_model('app/models/Mobilenet.h5')
            test_image = image.load_img(path, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image /=255
            a = acc.iloc[m - 1, 1]

        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        preds = Plants[np.argmax(result)]

        if preds == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
            msg = "Foliar fungicides can be used to manage gray leaf spot outbreaks"

        elif preds == "Corn_(maize)___Common_rust_":
            msg = "Use resistant varieties like DHM 103, Ganga Safed - 2 and avoid sowing of suceptable varieties like DHM 105"

        elif preds == "Corn_(maize)___healthy":
            msg = "Plant is Good no treatment required"

        elif preds == "Corn_(maize)___Northern_Leaf_Blight":
            msg = "Integration of early sowing, seed treatment and foliar spray with Tilt 25 EC (propiconazole) was the best combination in controlling maydis leaf blight and increasing maize yield"

        elif preds == "Potato___Early_blight":
            msg = "Mancozeb and chlorothalonil are perhaps the most frequently used protectant fungicides for early blight management"

        elif preds == "Potato___healthy":
            msg = "Plant is Good no treatment required"

        elif preds == "Potato___Late_blight":
            msg = "Effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days"

        elif preds == "Tomato___Bacterial_spot":
            msg = "When possible, is the best way to avoid bacterial spot on tomato. Avoiding sprinkler irrigation and cull piles near greenhouse or field operations, and rotating with a nonhost crop also helps control the disease"

        elif preds == "Tomato___healthy":
            msg = "Plant is Good no treatment required"

        elif preds == "Tomato___Late_blight":
            msg = "Ungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper can help protect plants from late tomato blight"

        else:
            msg = "Homemade Epsom salt mixture. Combine two tablespoons of Epsom salt with a gallon of water and spray the mixture on the plant"
        
    

        path1 = f'/static/img/{file.name}'
        print(path1)
        return render(request, "plantprediction.html", {'text': preds, 'a': round(a*100, 3), 'path': path1 ,'msg': msg})

    return render(request, 'plantprediction.html')
