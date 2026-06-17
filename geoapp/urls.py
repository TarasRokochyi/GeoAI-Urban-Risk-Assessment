from django.urls import path
from . import views

urlpatterns = [
    path("", views.map_view, name="map"),
    path("api/pois/", views.pois_geojson, name="pois-geojson"),
    path("api/predict/", views.predict, name="predict"),
    path("api/risk-zones/", views.risk_zones_geojson, name="risk-zones-geojson"),
]
