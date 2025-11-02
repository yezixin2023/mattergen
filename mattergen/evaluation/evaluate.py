# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import re
from pathlib import Path

import numpy as np
from pymatgen.core.structure import Structure

from mattergen.common.utils.globals import get_device
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.energy import EnergyMetricsCapability
from mattergen.evaluation.metrics.structure import StructureMetricsCapability
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)


def evaluate(
    structures: list[Structure],
    relax: bool = True,
    energies: list[float] | None = None,
    reference: ReferenceDataset | None = None,
    structure_matcher: (
        OrderedStructureMatcher | DisorderedStructureMatcher
    ) = DefaultDisorderedStructureMatcher(),
    save_as: str | None = None,
    potential_load_path: str | None = None,
    device: str = str(get_device()),
    structures_output_path: str | None = None,
    filtered_structures_path: str | None = None,
) -> dict[str, float | int]:
    """Evaluate the structures against a reference dataset.

    Args:
        structures: List of structures to evaluate.
        relax: Whether to relax the structures before evaluation. Note that if this is run,
            `energies` will be ignored.
        energies: Energies of the structures if already relaxed and computed externally
            (e.g., from DFT).
        reference: Reference dataset. If this is None, the default reference dataset will be
            used.
        structure_matcher: Structure matcher to use for matching the structures.
        save_as: Save the metrics as a JSON file.
        potential_load_path: Path to the Machine Learning potential to use for relaxation.
        device: Device to use for relaxation.
        structures_output_path: Path to save the relaxed structures.
        filtered_structures_path: Optional path to a directory where relaxed structures that
            are simultaneously stable, unique, and novel will be stored as CIF files along
            with a metadata JSON file containing their energies.

    Returns:
        metrics: a dictionary of metrics and their values.
    """
    if relax and energies is not None:
        raise ValueError("Cannot accept energies if relax is True.")
    if relax:
        relaxed_structures, energies = relax_structures(
            structures,
            device=device,
            potential_load_path=potential_load_path,
            output_path=structures_output_path,
        )
    else:
        relaxed_structures = structures
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=structures,
        reference=reference,
        structure_matcher=structure_matcher,
    )
    metrics = evaluator.compute_metrics(
        metrics=evaluator.available_metrics,
        save_as=save_as,
        pretty_print=True,
    )
    if filtered_structures_path is not None:
        _save_filtered_structures(
            evaluator=evaluator,
            structures=relaxed_structures,
            energies=energies,
            output_path=filtered_structures_path,
        )
    return metrics


def _save_filtered_structures(
    evaluator: MetricsEvaluator,
    structures: list[Structure],
    energies: list[float] | np.ndarray | None,
    output_path: str,
) -> None:
    """Persist filtered structures as CIF files and metadata."""

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(structures) == 0:
        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w") as handle:
            json.dump([], handle, indent=2)
        logger.info(
            "No structures to save after filtering; created empty metadata file at %s.",
            metadata_path,
        )
        return

    required_caps = {StructureMetricsCapability, EnergyMetricsCapability}
    if not required_caps.issubset(evaluator.available_capability_types):
        missing = required_caps.difference(evaluator.available_capability_types)
        raise ValueError(
            "Energy and structure capabilities are required to save filtered structures. "
            f"Missing: {missing}"
        )

    mask = evaluator.is_unique & evaluator.is_novel & evaluator.is_stable
    indices = np.where(mask)[0].tolist()

    filtered_structures = MetricsEvaluator.filter(structures, mask)
    filtered_energies: list[float] | None = None
    if energies is not None:
        energy_array = np.array(energies)
        filtered_energies = energy_array[mask].astype(float).tolist()

    energy_above_hull = evaluator.energy_capability.energy_above_hull

    payload = []
    used_filenames: set[str] = set()
    for filtered_idx, structure_idx in enumerate(indices):
        structure = filtered_structures[filtered_idx]
        cif_filename = _build_cif_filename(structure, structure_idx, used_filenames)
        used_filenames.add(cif_filename)
        cif_path = output_dir / cif_filename
        cif_path.write_text(structure.to(fmt="cif"))

        entry: dict[str, object] = {
            "structure_index": int(structure_idx),
            "reduced_formula": structure.composition.reduced_formula,
            "is_unique": bool(evaluator.is_unique[structure_idx]),
            "is_novel": bool(evaluator.is_novel[structure_idx]),
            "is_stable": bool(evaluator.is_stable[structure_idx]),
            "energy_above_hull": float(energy_above_hull[structure_idx]),
            "cif_file": cif_filename,
        }
        entry["total_energy"] = (
            float(filtered_energies[filtered_idx])
            if filtered_energies is not None
            else None
        )
        payload.append(entry)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(
        "Saved %s stable, unique, and novel structures to %s (metadata at %s)",
        len(payload),
        output_dir,
        metadata_path,
    )


def _build_cif_filename(
    structure: Structure, structure_idx: int, used_filenames: set[str]
) -> str:
    """Construct a unique CIF filename for the given structure."""

    formula = structure.composition.reduced_formula or "structure"
    safe_formula = re.sub(r"[^A-Za-z0-9]+", "_", formula).strip("_") or "structure"
    candidate = f"{safe_formula}_{structure_idx}.cif"
    counter = 1
    unique_candidate = candidate
    while unique_candidate in used_filenames:
        unique_candidate = f"{safe_formula}_{structure_idx}_{counter}.cif"
        counter += 1
    return unique_candidate
