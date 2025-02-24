from django.urls import path
from .views import home, predict_view


# Map the page to a URL
urlpatterns = [
    path('', home, name='home'), # This will load index.html
    path('predict/', predict_view, name='predict'), # API for predictions
]