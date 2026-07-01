"""
Builds a JSON lookup from each AMI-Traps GBIF accepted taxon key to the Quebec–Vermont model’s class
index, by matching species-level label strings in fgrained_labels.json to entries in the Quebec
category map, and saves it as taxon_to_quebec_idx.json for WebDataset conversion.
"""

import json

FGRAINED_LABELS_PATH = "~/data/ami_traps/insect_crops/fgrained_labels.json"
QUEBEC_CATEGORY_MAP_PATH = (
    "~/data/models/quebec-vermont_moth-category-map_19Jan2023.json"
)
TAXON_TO_QUEBEC_IDX_PATH = "~/data/fine_tuning_data/taxon_to_quebec_idx.json"

labels = json.load(open(FGRAINED_LABELS_PATH))
name_to_idx = json.load(open(QUEBEC_CATEGORY_MAP_PATH))

taxon_to_idx = {}
for meta in labels.values():
    if meta.get("taxon_rank") != "SPECIES":
        continue
    # include all regions if you expanded step 1; or filter meta["region"]
    name = meta["label"]
    key = str(meta["acceptedTaxonKey"])
    if name in name_to_idx and key != "-1":
        taxon_to_idx[key] = name_to_idx[name]

json.dump(
    taxon_to_idx,
    open(TAXON_TO_QUEBEC_IDX_PATH, "w"),
)
print(len(taxon_to_idx), "taxa mapped into Quebec head")
