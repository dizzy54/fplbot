from django import forms


class LastNameForm(forms.Form):
    last_name = forms.CharField(label='Player last name', max_length=100)
