from rest_framework.routers import DefaultRouter
from .views import ApiViewSet

router = DefaultRouter()
router.register(r'pics', ApiViewSet, basename='api')

urlpatterns = router.urls
