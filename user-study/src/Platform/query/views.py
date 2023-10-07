from os import TMP_MAX
from pydoc import cli
from django.shortcuts import render
from django.urls import reverse
from django.http import  HttpResponse, HttpResponseRedirect, Http404
from .models import QueryModel
from .trigger_test import send_trigger, trigger_dic, chinese_trigger_dic
import time
import os
import json

# 结果保存路径
SAVE_PATH = 'query/results/'
MOD = 1007
REST_PERIOD = 15
DOC_MOD = 47
FIRST_SCREEN_DOC = 4

model = QueryModel()

# homepage
def homepage(request):
    if request.POST.get('user_name'):
        user_id = str(request.POST['user_id'])
        user_name = request.POST['user_name']
        model.add_new_user(user_id, user_name)
        return HttpResponseRedirect(reverse('query:questions', args=(-1 + MOD, DOC_MOD, user_name, user_id)))
    else:
        return render(request, 'query/homepage.html')

# process function
def process_question_post(request, question_id, doc_id, user_name, user_id):

    if request.POST.get('endinfo') != None:
        end_info = request.POST.get('endinfo')
        model.add_info(user_id, question_id, end_info)
        print("end_info", end_info)

    if question_id == MOD - 1:
        question_id = -1
    ans = trigger_dic['start']
    if request.POST.get('grade') != None:
        ans = request.POST.get('grade')
        if ans in chinese_trigger_dic.keys():
            ans = chinese_trigger_dic[ans]
        else:
            ans = int(ans)
    click_doc_id = -1
    if request.POST.get('doc_id') != None:
        click_doc_id = int(request.POST.get('doc_id'))
    if question_id == model.get_question_numbers(user_id):
        return None
    # add rel info
    if request.POST.get('rel') != None:
        intent = request.POST.get('rel')
        model.add_intent(user_id, question_id, intent)
    
    rest = False
    if ans == trigger_dic['start']:
        question_id = question_id + 1
        doc_id = DOC_MOD
        # 每15个问题休息 不休息了？
        # if question_id % REST_PERIOD == 0 and question_id != 0:
        #     rest = True
    elif ans == trigger_dic['resume']:  # 从休息页面返回
        question_id = question_id
        doc_id = DOC_MOD
    elif ans == trigger_dic['click'] or ans == trigger_dic['back']:
        question_id = question_id 
        click_doc_id = click_doc_id
        doc_id = doc_id
    elif ans == trigger_dic['abandon']:
        question_id = question_id
        doc_id = (doc_id + 1) % (DOC_MOD + 1)
    elif ans == trigger_dic['end']:
        question_id = question_id
        doc_id = doc_id
    elif ans == trigger_dic['history']:
        question_id = question_id
        doc_id = doc_id
    elif ans == trigger_dic['first_screen']:
        question_id = question_id
        doc_id = (doc_id + 1) % (DOC_MOD + 1) + FIRST_SCREEN_DOC - 1
    elif ans == trigger_dic['lucky']:
        question_id = question_id
        doc_id = doc_id
        click_doc_id = 0
        # 显示end页面？
    print('question_id')
    time.sleep(0.1)
    send_trigger(question_id+90)
    print('doc_id')
    time.sleep(0.1)
    send_trigger(doc_id+20)
    
    model.add_action(user_id, {"quesion_id":question_id, "doc_id":doc_id, "ans":ans,"time_stamp":time.time()})
    
    if question_id == model.get_question_numbers(user_id):
        print("ans")
        time.sleep(0.1)
        send_trigger(ans)
        return None, question_id, doc_id

    # 返回下一个问题的名字，图片文件名，图片描述 
    if click_doc_id == -1:
        query, doc_info = model.get_question(user_id, question_id, doc_id)
    else:
        query, doc_info = model.get_question(user_id, question_id, click_doc_id)

    img1 = doc_info['crop_path'] if doc_info != None else None
    content = {
        'query': query,
        'landing_page': '',
        'crop': '',
        'ans': ans,
        'question_id': question_id,
        'user_name': user_name,
        'user_id': user_id,
        'doc_id': doc_id,
        'show_abandon': 1,
        'show_cross': 1,
        'rest': rest,
        'show_history': 0,
    }

    if doc_id == model.get_question_len(user_id, question_id) - 1 or doc_id == DOC_MOD - 2:
        content['show_abandon'] = 0
    if ans == trigger_dic['back'] or ans == trigger_dic['history']:
        content['show_cross'] = 0
    if ans == trigger_dic['history']:
        content['show_history'] = 1  
    if ans == trigger_dic['click'] or ans == trigger_dic['lucky']:
        img1 = doc_info['landing_page_path']
    elif ans == trigger_dic['end'] or ans == trigger_dic['history'] or ans == trigger_dic['first_screen']:
        img_list = []
        for tmp_doc_id in range(0, min(doc_id + 1, model.get_question_len(user_id, question_id))):
            tmp_query, tmp_img = model.get_question(user_id, question_id, tmp_doc_id)
            img_list.append(tmp_img['crop_path'])
        content['img_list'] = img_list
    if img1 != None:
        content['landing_page'] += img1
    if ans == trigger_dic['start']:
        content['intent'] = model.get_intent(user_id, question_id)
    print("ans ", ans)
    time.sleep(0.1)
    send_trigger(ans)
    return content, question_id, doc_id

def questions(request, question_id, doc_id, user_name, user_id):
    if not model.has_user(user_id):
        raise Http404("User does not exist!")
    if (question_id < 0 or question_id > model.get_question_numbers(user_id)) and question_id < MOD / 2:
        raise Http404("Question does not exist!")

    content, question_id, doc_id = process_question_post(request, question_id, doc_id, user_name, user_id)
    if content == None:
        model.add_info(user_id, -1, str(model.user_data[user_id]['action_list']))
        return HttpResponseRedirect(reverse('query:thanks'))
    elif content['rest']:
        return render(request, 'query/rest.html', content)
    elif content['ans'] == trigger_dic['start'] or content['ans'] == trigger_dic['resume']:
        return render(request, 'query/start.html', content)
    elif content['ans'] == trigger_dic['click']:
        return render(request, 'query/item.html', content)
    elif content['ans'] == trigger_dic['end']:
        return render(request, 'query/end.html', content)
    elif content['ans'] == trigger_dic['abandon'] or content['ans'] == trigger_dic['back'] or content['ans'] == trigger_dic['history']:
        return render(request, 'query/crop.html', content)
    elif content['ans'] == trigger_dic['first_screen']:
        return render(request, 'query/crop_first.html', content)
    elif content['ans'] == trigger_dic['lucky']:
        return render(request, 'query/item.html', content)
    raise Http404("ERROR!")

# final page
def thanks(request):
    return render(request, 'query/thanks.html', {})

# save audio
def save_audio(request):
    f1 = request.FILES.get('upfile')
    if f1 != None:
        # 文件保存路径
        fname = f1.name.split('_')
        if os.path.exists('../../user_data/' + fname[0] + '/record/') == False:
            os.mkdir('../../user_data/' + fname[0] + '/record/')
        fname = '../../user_data/' + fname[0] + '/record/' + fname[1] + '.mp3'
        with open(fname, 'wb') as pic:
            for c in f1.chunks():
                pic.write(c)
            
    return HttpResponse('上传成功')
