import os
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import Distance

from geoapp.models import POI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

CENTER_LAT = 38.7169
CENTER_LON = -9.1393
LAT_M = 111319.5
LON_M = 111319.5 * math.cos(math.radians(CENTER_LAT))

FEATURE_NAMES = [
    "dist_center_m",
    "poi_500m",
    "hosp_500m",
    "dist_hosp_m",
    "dist_police_m",
    "dist_fire_m",
]


def dist_m(lat1, lon1, lat2, lon2):
    return math.sqrt(((lat1 - lat2) * LAT_M) ** 2 + ((lon1 - lon2) * LON_M) ** 2)


def spatial_blocked_cv(X, coords, n_blocks=5):
    """Blocked CV по широті — запобігає просторовій автокореляції."""
    lats = coords[:, 0]
    q = np.linspace(lats.min(), lats.max(), n_blocks + 1)
    folds = []
    for i in range(n_blocks):
        lo, hi = q[i], q[i + 1]
        test_mask = (lats >= lo) & (lats <= hi)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


class Command(BaseCommand):
    help = "Навчити Random Forest на просторових ознаках POI та зберегти модель"

    def handle(self, *args, **options):
        pois = list(POI.objects.all())
        if len(pois) < 30:
            self.stderr.write(
                "Забагато мало POI. Спочатку запустіть: python manage.py import_osm"
            )
            return

        self.stdout.write(f"Обробляємо {len(pois)} POI...")

        # ── Будуємо DataFrame ────────────────────────────────────────────────
        records = []
        for p in pois:
            lat, lon = p.location.y, p.location.x
            pt = Point(lon, lat, srid=4326)

            poi_500m = POI.objects.filter(
                location__distance_lte=(pt, Distance(m=500))
            ).count() - 1  # -1 = себе

            hosp_500m = POI.objects.filter(
                category="hospital",
                location__distance_lte=(pt, Distance(m=500)),
            ).count()

            records.append({
                "lat": lat,
                "lon": lon,
                "category": p.category,
                "dist_center_m": dist_m(lat, lon, CENTER_LAT, CENTER_LON),
                "poi_500m": max(poi_500m, 0),
                "hosp_500m": hosp_500m,
            })

        df = pd.DataFrame(records)

        # відстань до найближчих служб
        hosp = df[df.category == "hospital"][["lat", "lon"]].values
        police = df[df.category == "police"][["lat", "lon"]].values
        fire = df[df.category == "fire_station"][["lat", "lon"]].values

        def min_dist(row, targets, default=5000.0):
            if len(targets) == 0:
                return default
            dists = [dist_m(row.lat, row.lon, t[0], t[1]) for t in targets]
            return min(dists)

        df["dist_hosp_m"] = df.apply(lambda r: min_dist(r, hosp), axis=1)
        df["dist_police_m"] = df.apply(lambda r: min_dist(r, police), axis=1)
        df["dist_fire_m"] = df.apply(lambda r: min_dist(r, fire), axis=1)

        # ── Синтетичний цільовий показник ────────────────────────────────────
        # Risk = зважена комбінація доступності екстрених служб
        score = (
            df["dist_hosp_m"] / df["dist_hosp_m"].max() * 0.40
            + df["dist_police_m"] / df["dist_police_m"].max() * 0.30
            + df["dist_fire_m"] / df["dist_fire_m"].max() * 0.20
            + (1 - df["poi_500m"] / (df["poi_500m"].max() + 1)) * 0.10
        )
        df["risk"] = pd.cut(score, bins=3, labels=["Low", "Medium", "High"])
        df = df.dropna(subset=["risk"])

        X = df[FEATURE_NAMES].values
        le = LabelEncoder()
        y = le.fit_transform(df["risk"])
        coords = df[["lat", "lon"]].values

        # ── Spatial Blocked CV ───────────────────────────────────────────────
        self.stdout.write("Spatial Blocked CV (5 lat-блоків)...")
        folds = spatial_blocked_cv(X, coords, n_blocks=5)
        cv_scores = []
        for train_idx, test_idx in folds:
            rf_cv = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            )
            rf_cv.fit(X[train_idx], y[train_idx])
            cv_scores.append(accuracy_score(y[test_idx], rf_cv.predict(X[test_idx])))

        self.stdout.write(
            f"  CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
        )

        # ── Фінальна модель ──────────────────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)

        self.stdout.write(f"Test accuracy: {acc:.4f}")
        self.stdout.write(classification_report(y_te, y_pred, target_names=le.classes_))

        # ── Оновлюємо ризик у базі ───────────────────────────────────────────
        self.stdout.write("Оновлюємо risk_level у базі...")
        X_all = df[FEATURE_NAMES].values
        risks = le.inverse_transform(rf.predict(X_all))
        probs = rf.predict_proba(X_all).max(axis=1)
        for i, p in enumerate(pois):
            if i < len(risks):
                p.risk_level = risks[i]
                p.risk_confidence = round(float(probs[i]), 3)
        POI.objects.bulk_update(pois[: len(risks)], ["risk_level", "risk_confidence"])

        # ── Зберігаємо модель ────────────────────────────────────────────────
        ml_dir = settings.ML_DIR
        ml_dir.mkdir(exist_ok=True)
        joblib.dump(rf, ml_dir / "rf_model.joblib")
        joblib.dump(le, ml_dir / "label_encoder.joblib")
        joblib.dump(FEATURE_NAMES, ml_dir / "feature_names.joblib")

        size_kb = (ml_dir / "rf_model.joblib").stat().st_size // 1024
        self.stdout.write(
            self.style.SUCCESS(
                f"Модель збережена: ml/rf_model.joblib ({size_kb} KB)"
            )
        )
