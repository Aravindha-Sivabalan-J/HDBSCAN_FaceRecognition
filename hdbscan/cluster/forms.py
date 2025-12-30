from django import forms
from .models import UploadedVideo

class EnrollmentForm(forms.Form):
    person_id = forms.CharField(label='Person ID (Name)', max_length=100)
    image = forms.ImageField(label='Reference Image')

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video_file', 'title']
