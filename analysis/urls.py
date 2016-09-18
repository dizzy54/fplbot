from django.conf.urls import url

from .views import PredictView, predict_score

urlpatterns = [
    url(r'^predict_fb/', PredictView.as_view()),
    url(r'^predict_score/', predict_score),
    # url(r'prediction_result/', prediction_result)
]
