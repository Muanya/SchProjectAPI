from rest_framework.parsers import JSONParser
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from projAPI.models import EmotionLoader, is_base64


class ApiViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['post'], url_path='emotion')
    def get_emotion(self, req):
        data = JSONParser().parse(req)
        try:
            image = data['image']
        except Exception as e:
            image = None

        if image:
            # verify base 64
            correct_encoding = is_base64(image)
            if not correct_encoding:
                return Response({'error': 'Image must be base64 encoded string or byte'},
                                status=status.HTTP_400_BAD_REQUEST)
            result = EmotionLoader(image).decode_image()
            return Response(result, status=status.HTTP_200_OK)
        return Response({'error': 'No image field'}, status=status.HTTP_400_BAD_REQUEST)
