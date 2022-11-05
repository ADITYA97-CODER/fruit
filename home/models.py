from django.db import models

# Create your models here.
class image(models.Model):
  image = models.ImageField(upload_to ='images',blank= True,null = True)