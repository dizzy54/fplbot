# from django.shortcuts import render

import json
import random
import traceback

from django.views import generic
from django.http.response import HttpResponse

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# from django.shortcuts import redirect
from django.conf import settings

import fb
from lib_1.prediction import predict

from django.shortcuts import render
from django.http import HttpResponseRedirect

from .forms import LastNameForm

from ratelimit.decorators import ratelimit


def predict_score(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = LastNameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            data = form.cleaned_data
            last_name_str = data['last_name']
            responses = get_response_from_last_name(last_name_str)
            responses_json = json.dumps(responses)
            return HttpResponse(responses_json)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = LastNameForm()

    return render(request, 'analysis/name.html', {'form': form})


def prediction_result(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = LastNameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = LastNameForm()

    return render(request, 'analysis/name.html', {'form': form})

PAGE_ACCESS_TOKEN = settings.PAGE_ACCESS_TOKEN
VERIFY_TOKEN = settings.VERIFY_TOKEN


class PredictView(generic.View):
    """View to handle Player A score prediction queries
    """
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return generic.View.dispatch(self, request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        # print self.request
        if self.request.GET['hub.verify_token'] == VERIFY_TOKEN:
            # basic inital setup here
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')

    @ratelimit(key='post:object', rate='10/m', method='POST', block=True)
    def post(self, request, *args, **kwargs):
        try:
            print "Handling Messages"
            payload = request.body
            print request.POST['object']
            # print payload
            data = json.loads(payload)
            messaging_entries = data["entry"][0]
            if "messaging" in messaging_entries and "message" in messaging_entries["messaging"][0]:
                for sender, message, respond in self.messaging_events(messaging_entries):
                    if respond:
                        print "Incoming from %s: %s" % (sender, message)
                        # user_details = fb.get_user_details(sender)
                        responses = self.get_responses_from_message(message)

                        # # send responses
                        for response in responses:
                            fb.send_message(sender, response)
        except:
            traceback.print_exc()
            message = 'Sorry. I got an unexpected error. Please try again later, or report to fplpredictor@gmail.com'
            return HttpResponse(message)
        return HttpResponse()

    def messaging_events(self, entries):
        """Generate tuples of (sender_id, message_text) from the
        provided payload.
        """
        # data = json.loads(payload)
        messaging_events = entries["messaging"]
        for event in messaging_events:
            if "message" in event and "text" in event["message"]:
                yield event["sender"]["id"], event["message"]["text"].encode('unicode_escape'), True
            else:
                yield event["sender"]["id"], "I can't respond to this", False

    def get_tagged_names(self, s):
        """returns list of tagged names from string
        """
        tag_indentifier = '#'
        words = s.split()
        names = []
        for word in words:
            if word[0] == tag_indentifier:
                names.append(word[1:])
        return names

    def get_responses_from_message(self, message):
        """returns list of responses
        """
        responses = []
        names = self.get_tagged_names(message)
        if len(names) == 0:
            # # no tagged name
            random_last_names = [
                'Mane',
                'Lukaku',
                'Costa',
                'Ibrahimovic',
                'Aguero',
                'Ozil',
                'Vardy',
                'Lamela',
                'Capoue'
            ]
            responses.append(
                """Sorry, I couldn't find any tagged names in your message.
                """
            )
            responses.append(
                """To get prediction for player, please include '#' followed by last name (eg. 'How much will #%s score?').
                """ % (
                    random.choice(random_last_names)
                )
            )
            responses.append(
                """
                Please don't blame me if predictions are innacurate. Feel free to send me feedback at fplpredictor@gmail.com.
                """
            )
        else:
            # # tagged names found
            for name in names:
                response = predict(name)
                responses = responses + response
        return responses


def limit_response(request, exception):
    message = "Sorry. Our server is too busy right now. Please try again after a few minutes."
    return HttpResponse(message, status=200)


def get_response_from_last_name(last_name):
    responses = []
    words = last_name.split()
    names = []
    for word in words:
        names.append(word)
    if len(names) == 0:
        responses.append("Please enter player's last name to get prediction.")
    else:
        # # tagged names found
        for name in names:
            # print "name=" + str(name)
            # print type(name)
            response = predict(name)
            responses = responses + response
    return responses
