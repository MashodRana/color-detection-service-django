from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from color_detection import views

urlpatterns = [
    path('color-detection/', views.color_detection_view, name='color-detection'),
    # path('color-detection-result/', v)
]
