from django.conf.urls import url

from .views import PvpView

urlpatterns = [
    url(r'^pvp/', PvpView.asView()),
]
