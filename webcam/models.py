from django.db import models

# Create your models here.

class Image(models.Model):
    img = models.ImageField(upload_to='images/')
    name = models.CharField(max_length=100)

    def __str__(self) -> str:
        return super().__str__()
    
class Video(models.Model):
    img = models.ImageField(upload_to='video/')
    name = models.CharField(max_length=100)

    def __str__(self) -> str:
        return super().__str__()