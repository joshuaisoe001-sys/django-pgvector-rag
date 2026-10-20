from rest_framework.routers import DefaultRouter

from .views import CollectionViewSet, DocumentViewSet

router = DefaultRouter()
router.register("collections", CollectionViewSet, basename="collection")
router.register("documents", DocumentViewSet, basename="document")

urlpatterns = router.urls
