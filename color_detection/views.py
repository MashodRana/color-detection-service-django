import os
import json
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

from .models import Image, ResponseModel
from .forms import UploadImageForm

from color_detection.service.color_detection import ColorDetector
from color_detection.service.service import Service
import base64

# Create your views here.


def color_detection_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            init_obj = form.save(commit=False)
            init_obj.file_name = init_obj.image.name
            init_obj.save()
            file_path = init_obj.image.url
            _path = settings.BASE_DIR + '/' + file_path
            print(_path)

            ser_obj = Service(image_path=_path)
            colors, percentage, fpath = ser_obj(ColorDetector(ser_obj.image))
            print(fpath)
            with open(fpath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

           # Save result in the response model 
            response = ResponseModel()
            response.file_name = fpath
            response.results = json.dumps({
                "colors":list(colors),
                "percentage": list(percentage)
            })
            response.save()

            # Create context dictionary which will be passed on html page.
            context = {
                'colors_name':colors,
                'colors_percentage': percentage,
                'image': img_data,
            }

            return render(request, 'color_detection/color_detection_result.html', context=context)
    else:
        form = UploadImageForm()
    return render(request, 'color_detection/color_detection.html', {'form': form})
