from django.shortcuts import render
from rest_framework import viewsets


class Testing:
    a = 5
    b = 8

    def __init__(self):
        pass

    def main(self):
        print(self.a)


class ApiView(viewsets.ViewSet):
    @staticmethod
    def trial():
        print('test')
    print('hass')
    pass
