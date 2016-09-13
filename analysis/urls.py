from django.conf.urls import url

from .views import PredictView

urlpatterns = [
    url(r'^predict/', PredictView.asView()),
]
