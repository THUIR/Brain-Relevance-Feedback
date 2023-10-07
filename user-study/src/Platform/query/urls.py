from django.urls import path
from . import views

app_name = 'query'
urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('q_id=<int:question_id>&d_id=<int:doc_id>&name=<str:user_name>&user_id=<str:user_id>', views.questions, name='questions'),
    path('thanks/', views.thanks, name='thanks'),
    path('save_audio/', views.save_audio),
]
