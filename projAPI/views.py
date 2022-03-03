from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response


class ApiViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['post'])
    def emotion(self, req):
        return Response({'key': 'success'})
