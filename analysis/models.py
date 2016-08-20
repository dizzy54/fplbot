from __future__ import unicode_literals

from django.db import models


class Player(models.Model):
    """Class to hold all player attributes
    """
    fpl_id = models.IntegerField('player id in fpl db')
    first_name = models.CharField('player first name', max_length=50)
    last_name = models.CharField('payer last name', max_length=50)
