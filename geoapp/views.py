import json
import os
import math
import numpy as np
import joblib
import shap
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import Distance
from django.contrib.gis.db.models.functions import Distance as DistanceFunc

from .models import POI, RiskZone

ML_DIR = settings.ML_DIR

# --- кешуємо модель і explainer в пам'яті ---
_rf = None
_le = None
_explainer = None
_feature_names = None


def _load_model():
    global _rf, _le, _explainer, _feature_names
    if _rf is None:
        rf_path = ML_DIR / "rf_model.joblib"
        le_path = ML_DIR / "label_encoder.joblib"
        fn_path = ML_DIR / "feature_names.joblib"
        if rf_path.exists():
            _rf = joblib.load(rf_path)
            _le = joblib.load(le_path)
            _feature_names = joblib.load(fn_path)
            _explainer = shap.TreeExplainer(_rf)
    return _rf, _le, _explainer, _feature_names


# Lisbon city center
CENTER_LON = -9.1393
CENTER_LAT = 38.7169
LAT_M = 111319.5
LON_M = 111319.5 * math.cos(math.radians(CENTER_LAT))


def _deg_to_m(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * LAT_M) ** 2 + ((lon1 - lon2) * LON_M) ** 2)


def _build_features(lat, lon):
    """Обчислює просторові ознаки для точки (lat, lon)."""
    pt = Point(lon, lat, srid=4326)

    dist_center_m = _deg_to_m(lat, lon, CENTER_LAT, CENTER_LON)

    # кількість POI в радіусі 500м (geography=True → метри)
    poi_500m = POI.objects.filter(
        location__distance_lte=(pt, Distance(m=500))
    ).count()

    # кількість лікарень у 500м
    hosp_500m = POI.objects.filter(
        category="hospital",
        location__distance_lte=(pt, Distance(m=500)),
    ).count()

    # відстань до найближчої лікарні
    nearest_h = (
        POI.objects.filter(category="hospital")
        .annotate(d=DistanceFunc("location", pt))
        .order_by("d")
        .first()
    )
    dist_hosp_m = nearest_h.d.m if nearest_h else 5000.0

    # відстань до найближчого поліцейського відділку
    nearest_p = (
        POI.objects.filter(category="police")
        .annotate(d=DistanceFunc("location", pt))
        .order_by("d")
        .first()
    )
    dist_police_m = nearest_p.d.m if nearest_p else 5000.0

    # відстань до найближчої пожежної частини
    nearest_f = (
        POI.objects.filter(category="fire_station")
        .annotate(d=DistanceFunc("location", pt))
        .order_by("d")
        .first()
    )
    dist_fire_m = nearest_f.d.m if nearest_f else 5000.0

    return np.array([[
        dist_center_m,
        poi_500m,
        hosp_500m,
        dist_hosp_m,
        dist_police_m,
        dist_fire_m,
    ]])


def map_view(request):
    poi_count = POI.objects.count()
    model_ready = (ML_DIR / "rf_model.joblib").exists()
    return render(request, "geoapp/map.html", {
        "poi_count": poi_count,
        "model_ready": model_ready,
    })


def pois_geojson(request):
    """GeoJSON всіх POI для Leaflet."""
    pois = POI.objects.all().order_by("id")[:1000]
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p.location.x, p.location.y]},
            "properties": {
                "id": p.id,
                "name": p.name or "",
                "category": p.category,
                "risk_level": p.risk_level,
                "risk_confidence": p.risk_confidence,
            },
        }
        for p in pois
    ]
    return JsonResponse({"type": "FeatureCollection", "features": features})


def risk_zones_geojson(request):
    """GeoJSON зон ризику для Split Map."""
    zones = RiskZone.objects.all()
    features = [
        {
            "type": "Feature",
            "geometry": json.loads(z.area.geojson),
            "properties": {"name": z.name, "risk_level": z.risk_level},
        }
        for z in zones
    ]
    return JsonResponse({"type": "FeatureCollection", "features": features})


@csrf_exempt
@require_POST
def predict(request):
    """
    GeoXAI Workflow:
    1. Отримує (lat, lon) з кліку
    2. Sampling ознак через PostGIS
    3. Inference (Random Forest)
    4. SHAP — пояснення прогнозу
    """
    try:
        data = json.loads(request.body)
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "Потрібні поля lat і lon"}, status=400)

    rf, le, explainer, feature_names = _load_model()
    if rf is None:
        return JsonResponse(
            {"error": "Модель не натренована. Запустіть: python manage.py train_model"},
            status=503,
        )

    X = _build_features(lat, lon)
    pred_idx = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0]
    risk = le.inverse_transform([pred_idx])[0]
    confidence = float(proba.max())

    # SHAP values (TreeExplainer — швидкий)
    shap_raw = explainer.shap_values(X)
    if isinstance(shap_raw, list):
        sv = shap_raw[pred_idx][0]
    else:
        sv = shap_raw[0, :, pred_idx]

    shap_dict = {name: round(float(v), 4) for name, v in zip(feature_names, sv)}

    return JsonResponse({
        "risk": risk,
        "confidence": round(confidence, 3),
        "probabilities": {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)},
        "shap": shap_dict,
        "features": {name: round(float(v), 2) for name, v in zip(feature_names, X[0])},
        "nearby_poi_count": int(X[0][1]),
    })
