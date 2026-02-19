"""
URL configuration for Simulation Based Imaging backend.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("api/", include("api.urls")),
    path("", TemplateView.as_view(template_name="index.html"), name="home"),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.FRONTEND_DIR / "assets")
    # Serve videos, images, and voxels from frontend public directory
    public_dir = settings.FRONTEND_DIR.parent / "public"
    urlpatterns += static("/videos/", document_root=public_dir / "videos")
    urlpatterns += static("/images/", document_root=public_dir / "images")
    urlpatterns += static("/voxels/", document_root=public_dir / "voxels")
