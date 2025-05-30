from django.urls import path
from .views import home, predict_view, analyse_combined_view, analyse_dt_only_view


# Map the page to a URL
urlpatterns = [
    path('', home, name='home'), # This will load index.html
    path('predict/', predict_view, name='predict'), # API for predictions
    path('analyse/', analyse_combined_view, name='analyse'), # API for predictions
    path("analyse_dt_only/", analyse_dt_only_view, name='analyse_dt_only'),
]