from django.contrib.gis.db import models


class POI(models.Model):
    CATEGORY_CHOICES = [
        ("hospital", "Hospital"),
        ("school", "School"),
        ("cafe", "Cafe"),
        ("pharmacy", "Pharmacy"),
        ("police", "Police"),
        ("fire_station", "Fire Station"),
        ("parking", "Parking"),
        ("other", "Other"),
    ]
    RISK_CHOICES = [("Low", "Low"), ("Medium", "Medium"), ("High", "High")]

    osm_id = models.BigIntegerField(unique=True, null=True, blank=True)
    name = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default="other")
    location = models.PointField(srid=4326, geography=True)

    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES, blank=True)
    risk_confidence = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["category"])]

    def __str__(self):
        return f"{self.name or 'POI'} ({self.category})"


class RiskZone(models.Model):
    RISK_CHOICES = [("low", "Low"), ("medium", "Medium"), ("high", "High")]

    name = models.CharField(max_length=255)
    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES)
    area = models.PolygonField(srid=4326, geography=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["risk_level"])]

    def __str__(self):
        return f"{self.name} ({self.risk_level})"
