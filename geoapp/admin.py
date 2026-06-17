from django.contrib.gis import admin
from .models import POI, RiskZone


@admin.register(POI)
class POIAdmin(admin.OSMGeoAdmin):
    list_display = ["name", "category", "risk_level", "risk_confidence"]
    list_filter = ["category", "risk_level"]
    search_fields = ["name"]
    readonly_fields = ["risk_level", "risk_confidence"]
    default_zoom = 13
    default_lon = -9.1393
    default_lat = 38.7169


@admin.register(RiskZone)
class RiskZoneAdmin(admin.OSMGeoAdmin):
    list_display = ["name", "risk_level", "created_at"]
    list_filter = ["risk_level"]
    default_zoom = 12
    default_lon = -9.1393
    default_lat = 38.7169
