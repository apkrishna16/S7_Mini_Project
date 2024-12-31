from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path("judgement/<str:file_name>/", views.serve_file, name="serve_file"),
]