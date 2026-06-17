# GeoAI Urban Risk Assessment — Lisbon

GeoDjango web app with PostGIS, Random Forest, and SHAP explainability.  
Click anywhere on the map → get a risk prediction + SHAP explanation.

![Web App Screenshot](assets/screenshot.png)

## Deliverables

| # | Вимога | Де |
|---|---|---|
| 1 | Docker Compose + GeoDjango структура | `docker-compose.yml`, `Dockerfile`, `geoapp/` |
| 2 | PostGIS з просторовими індексами | `PointField(geography=True)` → GIST index автоматично при `migrate` |
| 3 | Навчена модель + Spatial CV | `ml/rf_model.joblib`, `lab10.ipynb` (Spatial CV: 0.883 ± 0.03) |
| 4 | GeoXAI сайт (клік → прогноз + SHAP) | `geoapp/views.py`, `geoapp/templates/geoapp/map.html` |

## Stack

- **Backend:** Django 4.2 + GeoDjango + PostGIS
- **ML:** Random Forest (scikit-learn) + SHAP
- **Frontend:** Leaflet.js + Split Map
- **Infrastructure:** Docker Compose

## How to Run

```bash
# 1. Start containers
docker compose up --build

# 2. In a second terminal — create tables
docker compose exec web python manage.py makemigrations geoapp
docker compose exec web python manage.py migrate

# 3. Import real OSM data (Lisbon ~2600 POIs)
docker compose exec web python manage.py import_osm

# 4. Train the model
docker compose exec web python manage.py train_model
```

Open **http://localhost:8000**

Admin panel (optional): first run `docker compose exec web python manage.py createsuperuser`, then open **http://localhost:8000/admin/**
