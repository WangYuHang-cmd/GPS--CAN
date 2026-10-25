# Map policy (OSM drive graph) v0.1

- Source: Geofabrik regional extracts (`manifests/map_tiles.csv` records region, fetch date, md5 per tile); Overpass only for tiles outside downloaded regions.
- Drivable ways: `highway` in {motorway, motorway_link, trunk, trunk_link, primary, primary_link, secondary, secondary_link, tertiary, tertiary_link, unclassified, residential, living_street, road}; excluded: `area=yes`, `motor_vehicle=no`, `motorcar=no`, `access` in {private, no}, service roads (osmnx "drive" semantics).
- Direction: `oneway` in {yes, true, 1} or `junction` in {roundabout, circular} -> forward only; `oneway` in {-1, reverse} -> backward only; else both.
- Graph: nodes = OSM node ids; edges = consecutive node pairs of a way (unsimplified). Connectivity only through shared node ids, so bridges and tunnels that do not share nodes are not connected.
- Turn restrictions: relations `type=restriction` with `from` way, `via` node, `to` way; `no_*` forbids the (from, to) edge pair at the via node, `only_*` restricts the successors of the `from` edge to the listed `to` edges. Way-via restrictions are ignored (rare).
- Tiles: 0.25 degree, a way is stored in every tile that any of its nodes falls in; de-duplicated by way id on load.
- Region handed to a method: tiles intersecting the disc around p0 with radius r(tau) = 5 km + 40 m/s * tau (tau = window duration for fixed windows; capped at 60 km in the pilot), never derived from the trajectory.
- Verified on Utah (2026-10-22 extract): 99.1% of 1 Hz reference points of a Salt Lake drive lie within 10 m of a drivable edge (median 1.5 m).
