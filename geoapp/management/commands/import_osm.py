import time
import requests
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from geoapp.models import POI

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Lisbon bounding box: south,west,north,east
BBOX = "38.65,-9.25,38.80,-9.05"

AMENITY_CATEGORIES = {
    "hospital": "hospital",
    "school": "school",
    "cafe": "cafe",
    "pharmacy": "pharmacy",
    "police": "police",
    "fire_station": "fire_station",
    "parking": "parking",
}


class Command(BaseCommand):
    help = "Імпортувати POI Лісабону з OpenStreetMap через Overpass API"

    def handle(self, *args, **options):
        self.stdout.write("Завантаження даних з OpenStreetMap...")
        total_created = 0

        for amenity, category in AMENITY_CATEGORIES.items():
            query = (
                f"[out:json][timeout:30];"
                f"node[amenity={amenity}]({BBOX});"
                f"out body;"
            )
            try:
                resp = requests.post(
                    OVERPASS_URL, data={"data": query}, timeout=35,
                    headers={"User-Agent": "GeoDjango-Lab10/1.0 (student project)"}
                )
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
            except Exception as e:
                self.stderr.write(f"  {amenity}: помилка запиту — {e}")
                continue

            created = 0
            for el in elements:
                if el.get("type") != "node":
                    continue
                tags = el.get("tags", {})
                _, was_created = POI.objects.get_or_create(
                    osm_id=el["id"],
                    defaults={
                        "name": tags.get("name", ""),
                        "category": category,
                        "location": Point(el["lon"], el["lat"], srid=4326),
                    },
                )
                if was_created:
                    created += 1

            total_created += created
            self.stdout.write(f"  {category:15s}: {len(elements):4d} знайдено, {created:4d} нових")
            time.sleep(1)  # щоб не перевантажувати Overpass

        self.stdout.write(
            self.style.SUCCESS(f"\nРезультат: {total_created} нових POI у базі")
        )
        self.stdout.write(f"Всього POI: {POI.objects.count()}")
