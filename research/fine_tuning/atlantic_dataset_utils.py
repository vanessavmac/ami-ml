"""Shared helpers for Atlantic Forestry fine-tuning dataset preparation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PSEUDO_SPECIES_LABELS = frozenset({"Not Identifiable", "Not Lepidoptera"})

# Species excluded from Atlantic fine-tuning (Antenna GBIF stub; single-sample edge case).
ATLANTIC_DATASET_EXCLUDED_SPECIES = frozenset({"Alcis porcelaria"})

DEFAULT_ANTENNA_API_BASE_URL = "https://api.antenna.insectai.org/api/v2"


class GbifKeyNotFoundError(Exception):
    """Raised when Antenna taxonomy has no GBIF key for a trainable taxon."""


class GbifKeyMismatchError(Exception):
    """Raised when Antenna GBIF key disagrees with AMI-Traps fgrained_labels."""


class GbifKeyConflictError(Exception):
    """Raised when fgrained_labels maps one species name to multiple acceptedTaxonKeys."""


ISSUE_RANK_NOT_SPECIES = "rank_not_species"
ISSUE_NULL_GBIF_TAXON_KEY = "null_gbif_taxon_key"
ISSUE_NEEDS_ANTENNA_FIX = "needs_antenna_fix"
ISSUE_USED_FGRAINED_FALLBACK = "used_fgrained_fallback"
ISSUE_USED_SYNONYM_OF_FALLBACK = "used_synonym_of_fallback"

RESOLUTION_ANTENNA_TAXON = "antenna_taxon"
RESOLUTION_OCCURRENCE_FALLBACK = "occurrence_fallback"
RESOLUTION_SYNONYM_OF = "synonym_of"
RESOLUTION_FGRAINED_LABELS = "fgrained_labels"

_ANTENNA_FIX_HINT = (
    "Exposing synonym_of_id on TaxonSerializer would enable synonym fallback."
)


@dataclass(frozen=True)
class GbifResolutionResult:
    gbif_taxon_key: str
    resolution_source: str
    determination_id: str
    species_name: str
    antenna_taxon_name: str | None
    antenna_rank: str | None
    antenna_gbif_taxon_key: str | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class AntennaTaxonRecord:
    taxon_id: str
    name: str
    rank: str
    gbif_taxon_key: str | None


def load_quebec_category_map(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fgrained_labels(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fgrained_species_to_accepted_key(
    fgrained_labels: dict[str, Any],
) -> dict[str, str]:
    """Map species label string to GBIF acceptedTaxonKey from AMI-Traps fgrained_labels."""
    mapping: dict[str, str] = {}
    for meta in fgrained_labels.values():
        if meta.get("taxon_rank") != "SPECIES":
            continue
        name = meta["label"]
        key = str(meta["acceptedTaxonKey"])
        if key == "-1":
            continue
        if name in mapping and mapping[name] != key:
            raise GbifKeyConflictError(
                f"fgrained_labels has conflicting acceptedTaxonKey for {name!r}: "
                f"{mapping[name]} vs {key}"
            )
        mapping[name] = key
    return mapping


def load_antenna_taxon_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_antenna_taxon_cache(
    path: str | Path, cache: dict[str, dict[str, Any]]
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


class AntennaTaxonClient:
    """Read-only client for Antenna taxonomy / occurrence API."""

    def __init__(
        self,
        base_url: str = DEFAULT_ANTENNA_API_BASE_URL,
        api_token: str | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token or os.environ.get("ANTENNA_API_TOKEN")
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = load_antenna_taxon_cache(cache_path) if cache_path else {}

    def _request_json(self, url: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Antenna API request failed ({exc.code}): {url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Antenna API request failed: {url}") from exc

    def get_taxon(self, taxon_id: int | str) -> dict[str, Any]:
        taxon_key = str(int(float(taxon_id)))
        if taxon_key in self._cache:
            return self._cache[taxon_key]
        url = f"{self.base_url}/taxa/{taxon_key}/"
        payload = self._request_json(url)
        self._cache[taxon_key] = payload
        if self.cache_path:
            save_antenna_taxon_cache(self.cache_path, self._cache)
        return payload

    def get_occurrence(self, occurrence_id: int | str) -> dict[str, Any]:
        occurrence_key = str(int(float(occurrence_id)))
        url = f"{self.base_url}/occurrences/{occurrence_key}/"
        return self._request_json(url)

    def search_taxa_by_name(
        self, name: str, rank: str = "SPECIES"
    ) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"name": name, "rank": rank, "limit": 20})
        url = f"{self.base_url}/taxa/?{params}"
        payload = self._request_json(url)
        return payload.get("results", [])

    def iter_all_taxa(self, page_size: int = 500) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = urllib.parse.urlencode({"limit": page_size, "offset": offset})
            url = f"{self.base_url}/taxa/?{params}"
            payload = self._request_json(url)
            batch = payload.get("results", [])
            if not batch:
                break
            results.extend(batch)
            if payload.get("next") is None:
                break
            offset += page_size
        return results


def _normalize_gbif_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(int(float(value)))


def _parse_taxon_record(
    taxon_id: int | str, payload: dict[str, Any]
) -> AntennaTaxonRecord:
    return AntennaTaxonRecord(
        taxon_id=str(int(float(taxon_id))),
        name=str(payload.get("name", "")),
        rank=str(payload.get("rank", "")),
        gbif_taxon_key=_normalize_gbif_key(payload.get("gbif_taxon_key")),
    )


def _gbif_key_from_taxon_payload(payload: dict[str, Any]) -> str | None:
    return _normalize_gbif_key(payload.get("gbif_taxon_key"))


def _gbif_key_from_occurrence(payload: dict[str, Any]) -> str | None:
    details = payload.get("determination_details") or {}
    taxon = details.get("taxon") or {}
    return _normalize_gbif_key(taxon.get("gbif_taxon_key"))


def _synonym_of_taxon_id(payload: dict[str, Any]) -> str | None:
    synonym_id = payload.get("synonym_of_id")
    if synonym_id is not None:
        return str(int(float(synonym_id)))
    synonym = payload.get("synonym_of")
    if isinstance(synonym, dict) and synonym.get("id") is not None:
        return str(int(float(synonym["id"])))
    return None


def _recommended_action(result: GbifResolutionResult) -> str:
    if ISSUE_NEEDS_ANTENNA_FIX not in result.issues:
        return "No action needed"
    taxon_id = result.determination_id
    return (
        f"Set rank=SPECIES and gbif_taxon_key on Antenna taxon {taxon_id}, "
        f"or set synonym_of to the accepted taxon. {_ANTENNA_FIX_HINT}"
    )


def _build_resolution_result(
    *,
    gbif_key: str,
    resolution_source: str,
    record: AntennaTaxonRecord,
    species_name: str,
    issues: list[str],
) -> GbifResolutionResult:
    if resolution_source != RESOLUTION_ANTENNA_TAXON:
        if ISSUE_NEEDS_ANTENNA_FIX not in issues:
            issues.append(ISSUE_NEEDS_ANTENNA_FIX)
    elif ISSUE_RANK_NOT_SPECIES in issues:
        if ISSUE_NEEDS_ANTENNA_FIX not in issues:
            issues.append(ISSUE_NEEDS_ANTENNA_FIX)
    return GbifResolutionResult(
        gbif_taxon_key=gbif_key,
        resolution_source=resolution_source,
        determination_id=record.taxon_id,
        species_name=species_name,
        antenna_taxon_name=record.name or None,
        antenna_rank=record.rank or None,
        antenna_gbif_taxon_key=record.gbif_taxon_key,
        issues=tuple(dict.fromkeys(issues)),
    )


def resolve_gbif_taxon_key_for_determination(
    client: AntennaTaxonClient,
    determination_id: int | str,
    *,
    species_name: str,
    sample_occurrence_id: int | str | None,
    fgrained_ref: dict[str, str],
) -> GbifResolutionResult:
    """Resolve GBIF acceptedTaxonKey via layered Antenna + fgrained fallbacks."""
    taxon_payload = client.get_taxon(determination_id)
    record = _parse_taxon_record(determination_id, taxon_payload)
    issues: list[str] = []

    if record.rank != "SPECIES":
        issues.append(ISSUE_RANK_NOT_SPECIES)
    if record.gbif_taxon_key is None:
        issues.append(ISSUE_NULL_GBIF_TAXON_KEY)

    gbif_key = record.gbif_taxon_key
    resolution_source = RESOLUTION_ANTENNA_TAXON

    if gbif_key is None and sample_occurrence_id is not None:
        occurrence_payload = client.get_occurrence(sample_occurrence_id)
        gbif_key = _gbif_key_from_occurrence(occurrence_payload)
        if gbif_key is not None:
            resolution_source = RESOLUTION_OCCURRENCE_FALLBACK

    if gbif_key is None:
        synonym_id = _synonym_of_taxon_id(taxon_payload)
        if synonym_id is not None:
            accepted_payload = client.get_taxon(synonym_id)
            gbif_key = _gbif_key_from_taxon_payload(accepted_payload)
            if gbif_key is not None:
                resolution_source = RESOLUTION_SYNONYM_OF
                issues.append(ISSUE_USED_SYNONYM_OF_FALLBACK)

    if gbif_key is None and species_name in fgrained_ref:
        gbif_key = fgrained_ref[species_name]
        resolution_source = RESOLUTION_FGRAINED_LABELS
        issues.append(ISSUE_USED_FGRAINED_FALLBACK)

    if gbif_key is None:
        raise GbifKeyNotFoundError(
            f"no gbif_taxon_key for determination_id {record.taxon_id} "
            f"({species_name!r}) after Antenna taxon, occurrence, synonym_of, "
            f"and fgrained fallbacks. Antenna taxon {record.taxon_id} may need "
            f"platform fix (rank/gbif_taxon_key/synonym_of). {_ANTENNA_FIX_HINT}"
        )

    if (
        resolution_source in (RESOLUTION_ANTENNA_TAXON, RESOLUTION_OCCURRENCE_FALLBACK)
        and species_name in fgrained_ref
        and gbif_key != fgrained_ref[species_name]
    ):
        raise GbifKeyMismatchError(
            f"GBIF key mismatch for {species_name!r}: antenna={gbif_key}, "
            f"fgrained_labels={fgrained_ref[species_name]}"
        )

    return _build_resolution_result(
        gbif_key=gbif_key,
        resolution_source=resolution_source,
        record=record,
        species_name=species_name,
        issues=issues,
    )


def issues_report_to_dataframe(
    results: list[GbifResolutionResult],
    fgrained_ref: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "determination_id": result.determination_id,
                "species_name": result.species_name,
                "antenna_taxon_name": result.antenna_taxon_name,
                "antenna_rank": result.antenna_rank,
                "antenna_gbif_taxon_key": result.antenna_gbif_taxon_key or "",
                "resolved_gbif_taxon_key": result.gbif_taxon_key,
                "resolution_source": result.resolution_source,
                "issues": ";".join(result.issues),
                "recommended_action": _recommended_action(result),
                "in_fgrained_labels": result.species_name in fgrained_ref,
            }
        )
    return pd.DataFrame(rows)


def write_antenna_taxon_issues_report(
    results: list[GbifResolutionResult],
    path: str | Path,
    fgrained_ref: dict[str, str],
) -> pd.DataFrame:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df = issues_report_to_dataframe(results, fgrained_ref)
    df.to_csv(report_path, index=False)
    return df


def print_gbif_resolution_summary(results: list[GbifResolutionResult]) -> None:
    by_source: dict[str, int] = {}
    needs_fix = 0
    for result in results:
        by_source[result.resolution_source] = (
            by_source.get(result.resolution_source, 0) + 1
        )
        if ISSUE_NEEDS_ANTENNA_FIX in result.issues:
            needs_fix += 1
    print("\n## GBIF key resolution summary", flush=True)
    print(f"- Species resolved: {len(results)}", flush=True)
    for source in (
        RESOLUTION_ANTENNA_TAXON,
        RESOLUTION_OCCURRENCE_FALLBACK,
        RESOLUTION_SYNONYM_OF,
        RESOLUTION_FGRAINED_LABELS,
    ):
        if by_source.get(source):
            print(f"- {source}: {by_source[source]}", flush=True)
    print(f"- Antenna taxa needing platform fix: {needs_fix}", flush=True)


def resolve_gbif_taxon_keys_from_dataframe(
    trainable_df: pd.DataFrame,
    *,
    client: AntennaTaxonClient,
    fgrained_labels_path: str | Path,
    determination_id_col: str = "determination_id",
    species_col: str = "determination_name",
    occurrence_id_col: str = "id",
    issues_report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add gbif_taxon_key column; hard-fail on missing keys or overlap mismatches."""
    fgrained_ref = build_fgrained_species_to_accepted_key(
        load_fgrained_labels(fgrained_labels_path)
    )

    sample_occurrence_by_taxon: dict[str, str] = {}
    for _, row in trainable_df.iterrows():
        taxon_id = str(int(float(row[determination_id_col])))
        sample_occurrence_by_taxon.setdefault(
            taxon_id, str(int(float(row[occurrence_id_col])))
        )

    taxon_to_gbif: dict[str, str] = {}
    taxon_to_source: dict[str, str] = {}
    resolution_results: list[GbifResolutionResult] = []
    for _, row in trainable_df.drop_duplicates(
        subset=[determination_id_col]
    ).iterrows():
        taxon_id = str(int(float(row[determination_id_col])))
        species_name = row[species_col]
        result = resolve_gbif_taxon_key_for_determination(
            client,
            taxon_id,
            species_name=species_name,
            sample_occurrence_id=sample_occurrence_by_taxon.get(taxon_id),
            fgrained_ref=fgrained_ref,
        )
        resolution_results.append(result)
        taxon_to_gbif[taxon_id] = result.gbif_taxon_key
        taxon_to_source[taxon_id] = result.resolution_source

    if issues_report_path is not None:
        write_antenna_taxon_issues_report(
            resolution_results, issues_report_path, fgrained_ref
        )
    print_gbif_resolution_summary(resolution_results)

    out = trainable_df.copy()
    out["gbif_taxon_key"] = out[determination_id_col].apply(
        lambda x: taxon_to_gbif[str(int(float(x)))]
    )
    out["gbif_resolution_source"] = out[determination_id_col].apply(
        lambda x: taxon_to_source[str(int(float(x)))]
    )
    return out


def build_taxon_to_quebec_idx(
    fgrained_labels: dict[str, Any], name_to_idx: dict[str, int]
) -> dict[str, int]:
    """Map GBIF accepted taxon key (str) to Quebec class index."""
    taxon_to_idx: dict[str, int] = {}
    for meta in fgrained_labels.values():
        if meta.get("taxon_rank") != "SPECIES":
            continue
        name = meta["label"]
        key = str(meta["acceptedTaxonKey"])
        if name in name_to_idx and key != "-1":
            taxon_to_idx[key] = name_to_idx[name]
    return taxon_to_idx


def normalize_taxon_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(int(float(value)))


def is_binomial_species_name(name: str) -> bool:
    return isinstance(name, str) and " " in name.strip()


def filter_atlantic_trainable_rows(
    df: pd.DataFrame, quebec_map: dict[str, int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return verified species-level rows whose name exists in the Quebec map."""
    stats = {
        "total_rows": len(df),
        "not_verified": 0,
        "pseudo_label": 0,
        "non_binomial": 0,
        "not_in_quebec_map": 0,
        "excluded_species": 0,
        "trainable_rows": 0,
    }

    verified = df[df["verification_status"] == True].copy()  # noqa: E712
    stats["not_verified"] = stats["total_rows"] - len(verified)

    pseudo_mask = verified["determination_name"].isin(PSEUDO_SPECIES_LABELS)
    stats["pseudo_label"] = int(pseudo_mask.sum())
    verified = verified[~pseudo_mask]

    binomial_mask = verified["determination_name"].apply(is_binomial_species_name)
    stats["non_binomial"] = int((~binomial_mask).sum())
    verified = verified[binomial_mask]

    in_map_mask = verified["determination_name"].isin(quebec_map)
    stats["not_in_quebec_map"] = int((~in_map_mask).sum())
    trainable = verified[in_map_mask].copy()

    excluded_mask = trainable["determination_name"].isin(
        ATLANTIC_DATASET_EXCLUDED_SPECIES
    )
    stats["excluded_species"] = int(excluded_mask.sum())
    trainable = trainable[~excluded_mask].copy()
    stats["trainable_rows"] = len(trainable)

    return trainable, stats


def aggregate_species_counts(
    df: pd.DataFrame,
    species_col: str,
    taxon_col: str,
    quebec_map: dict[str, int],
) -> pd.DataFrame:
    """Aggregate image counts per species with GBIF taxon key and Quebec class index."""
    grouped = (
        df.groupby(species_col, as_index=False)
        .agg(n_images=(species_col, "size"), gbif_taxon_key=(taxon_col, "first"))
        .rename(columns={species_col: "species_name"})
    )
    grouped["gbif_taxon_key"] = grouped["gbif_taxon_key"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else None
    )
    grouped["quebec_class_idx"] = grouped["species_name"].map(quebec_map)
    return grouped.sort_values("n_images", ascending=False).reset_index(drop=True)
