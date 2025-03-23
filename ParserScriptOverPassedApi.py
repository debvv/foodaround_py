import overpy
import pandas as pd

api = overpy.Overpass()

# Кишинёв и его bounding box
bbox = "47.0,28.75,47.2,29.1"  # (south,west,north,east)

# Запрос на заведения общепита
query = f"""
    [out:json][timeout:60];
    (
      node["amenity"~"restaurant|cafe|fast_food|bar|pub"]( {bbox} );
      way["amenity"~"restaurant|cafe|fast_food|bar|pub"]( {bbox} );
      relation["amenity"~"restaurant|cafe|fast_food|bar|pub"]( {bbox} );
    );
    out center tags;
"""

result = api.query(query)

# Парсинг
places = []
for node in result.nodes + result.ways + result.relations:
    tags = node.tags
    places.append({
        "name": tags.get("name"),
        "type": tags.get("amenity"),
        "cuisine": tags.get("cuisine"),
        "address": tags.get("addr:full") or tags.get("addr:street"),
        "lat": node.lat if hasattr(node, 'lat') else node.center_lat,
        "lon": node.lon if hasattr(node, 'lon') else node.center_lon,
        "phone": tags.get("contact:phone") or tags.get("phone"),
        "website": tags.get("website"),
        "opening_hours": tags.get("opening_hours")
    })

# В датафрейм
df = pd.DataFrame(places)
df.to_csv("osm_chisinau_restaurants.csv", index=False)

print(f"Собрано: {len(df)} заведений")
