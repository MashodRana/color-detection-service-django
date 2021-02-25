from django.db import models
from django.core.files.storage import FileSystemStorage


# Create your models here.

def upload_here(instance, file_name):
    """ Upload file as per User """

    return f'{instance.user.id}/{file_name}' #  file will be uploaded to MEDIA_ROOT/user_id/file_name


class Image(models.Model):
    """ Model/Table for handling image through database. """
    file_name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='images/') # file will be uploaded at 'MEDIA_ROOT/images/'
    description = models.CharField(max_length=511, blank=True)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file_name

class ResponseModel(models.Model):
    """ Model/Table to store response """
    file_path = models.CharField(max_length=255) # output image which is pie chart of presnted color in the image
    results = models.CharField(max_length=255)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.results
