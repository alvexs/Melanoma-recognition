from django.db import models
from django.contrib.auth.models import User


class Image(models.Model):
    id = models.AutoField(auto_created=True, primary_key=True)
    file_obj = models.FileField(upload_to='images/')


class Mask(models.Model):
    id = models.AutoField(auto_created=True, primary_key=True)
    mask_url = models.CharField(max_length=1024, null=False)


class RecognitionResult(models.Model):
    id = models.AutoField(auto_created=True, primary_key=True)
    prediction = models.FloatField(null=False)
    featured_image_url = models.CharField(max_length=1024, null=False)


class Session(models.Model):
    id = models.AutoField(auto_created=True, primary_key=True)
    upload_image = models.ForeignKey(Image, on_delete=models.CASCADE)
    rec_result = models.ForeignKey(RecognitionResult, on_delete=models.CASCADE)
    mask = models.ForeignKey(Mask, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
