# from django.shortcuts import render

import json
import random

from django.views import generic
from django.http.response import HttpResponse

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# from django.shortcuts import redirect
from django.conf import settings

import fb
from . import lib

PAGE_ACCESS_TOKEN = settings.PAGE_ACCESS_TOKEN
VERIFY_TOKEN = settings.VERIFY_TOKEN


class PredictView(generic.View):
    """View to handle Player A score prediction queries
    """
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return generic.View.dispatch(self, request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        if self.request.GET['hub.verify_token'] == VERIFY_TOKEN:
            # basic inital setup here
            return HttpResponse(self.request.GET['hub.challenge'])
        else:
            return HttpResponse('Error, invalid token')

    def post(self, request, *args, **kwargs):
        print "Handling Messages"
        payload = request.body
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
                        fb.send_message(sender, message)

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
                 To get prediction for player, please include '#' followed by last name
                 (eg. 'How much will #%s score?').
                """ % (
                    random.choice(random_last_names)
                )
            )
            responses.append(
                """
                Please don't blame me if predictions are innacurate. I'm just a dumb messenger for maths :p.
                 You can send me your feedback at fplpredictor@gmail.com, I would love to get your opinion.
                """
            )
        else:
            # # tagged names found
            for name in names:
                response = lib.prediction.predict(name)
                responses = responses + response
        return responses
