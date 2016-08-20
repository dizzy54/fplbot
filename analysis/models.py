from __future__ import unicode_literals

from django.db import models


class Player(models.Model):
    """Class to hold all player attributes
    """
    fpl_id = models.IntegerField('player id in fpl db')
    first_name = models.IntegerField('player first name')
    last_name = models.IntegerField('payer last name')
