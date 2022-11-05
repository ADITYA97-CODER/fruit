from django.shortcuts import render
from home.models import image
import cv2
from keras.models import load_model
import numpy as np
model = load_model('fruit.h5')

def preprocess(img):
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY )
    img = cv2.equalizeHist(img)
    img= img/255
    return img
# Create your views here.
def home(request):
    name = ['apple','bannana','mixed','orange']
    if request.method == "POST":
        img = request.FILES['image']
        imag = image(image =img)
        imag.save()
        an = imag.image.name
        i= cv2.imread('./static/images/'+an)
        iag = cv2.resize(i , (32,32))
        iag = preprocess(iag)
        iag = np.asarray(iag)
        iag = iag.reshape(1,32,32,1)
        confidence =model.predict(iag)
    
        m='yes'
        no = np.argmax(model.predict(iag))
        
        context = {'name':name[no],'confidence':confidence ,'m':m}
        return render(request,'home.html',context)

    return render(request,'home.html')