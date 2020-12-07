from deployment.settings import BASE_DIR
from django.shortcuts import render
from .utils import predict
from django.core.files.storage import FileSystemStorage
import os
import base64


def landing_page(request):
    data = {}
    if request.method == "POST" and request.FILES['file']:
        # print(request.FILES)
        file = request.FILES['file']
        fs = FileSystemStorage(location='deploymodel/datatmp/')
        filename = fs.save(file.name, file)
        photoname = fs.generate_filename(filename)
        # print(os.path.abspath("./datatmp/"+photoname))
        data['message'] = predict(photoname)
        with open(str(BASE_DIR)+"/deploymodel"+'/datatmp/'+photoname, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            print(encoded_string)
            data['image'] = encoded_string.decode('utf-8')
    return render(request, "deploymodel/index.html", data)
