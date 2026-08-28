"""Frozen PLS adapter schema for the external complete-protein V4 features."""

from __future__ import annotations

import torch


PHYSCHEM_DIMENSION = 62
V4_SPATIAL_SCALAR_DIMENSION = 89
PLS_SPATIAL_SCALAR_DIMENSION = 90
SPATIAL_VECTOR_CHANNELS = 8

V4_SPATIAL_SCALAR_NAMES = (
    [f"v2_sasa_{i}" for i in range(6)] +
    [f"v2_surface_{i}" for i in range(14)] +
    [f"v2_structural_geometry_{i}" for i in range(26)] +
    [f"v2_enhanced_{i}" for i in range(12)] +
    [f"local_density_raw_{i}" for i in range(4)] +
    ["distance_to_center_raw", "surface_normal_magnitude"] +
    [f"charge_density_raw_{i}" for i in range(4)] +
    [f"hydrophobicity_{i}" for i in range(4)] +
    ["phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos"] +
    ["nearest_sidechain_dist", "packing_score", "void_ratio"] +
    ["env_aromatic_ratio", "env_hbond_donor_ratio", "env_hbond_acceptor_ratio", "env_gly_pro_ratio"] +
    ["self_metal_binder", "neighbor_metal_binder_6A", "neighbor_metal_binder_10A"] +
    ["local_anisotropy"]
)
PLS_SPATIAL_SCALAR_NAMES = V4_SPATIAL_SCALAR_NAMES + ["raw_plddt_fraction"]


def adapt_v4_features(features: dict, raw_plddt: torch.Tensor) -> dict:
    """Validate V4 tensors and append raw pLDDT/100 as the explicit 90th scalar."""
    physchem = features["physchem_features"].float()
    scalars = features["spatial_scalar_features"].float()
    vectors = features["spatial_vector_features"].float()
    coordinates = features["ca_coords"].float()
    residues = int(features["n_residues"])
    expected = {
        "physchem_features": (residues, PHYSCHEM_DIMENSION),
        "spatial_scalar_features": (residues, V4_SPATIAL_SCALAR_DIMENSION),
        "spatial_vector_features": (residues, SPATIAL_VECTOR_CHANNELS, 3),
        "ca_coords": (residues, 3), "raw_plddt": (residues,),
    }
    observed = {"physchem_features": tuple(physchem.shape), "spatial_scalar_features": tuple(scalars.shape),
                "spatial_vector_features": tuple(vectors.shape), "ca_coords": tuple(coordinates.shape),
                "raw_plddt": tuple(raw_plddt.shape)}
    if observed != expected:
        raise ValueError(f"V4 feature shape mismatch: expected={expected}, observed={observed}")
    if not all(torch.isfinite(value).all() for value in (physchem, scalars, vectors, coordinates, raw_plddt)):
        raise ValueError("V4 feature tensors must be finite")
    plddt = raw_plddt.float()
    if plddt.max() > 1.0:
        plddt = plddt / 100.0
    if torch.any((plddt < 0) | (plddt > 1)):
        raise ValueError("raw pLDDT must be in [0,1] or [0,100]")
    return {"physchem_features": physchem, "spatial_scalar_features": torch.cat([scalars, plddt[:, None]], 1),
            "spatial_vector_features": vectors, "ca_coords": coordinates,
            "plddt": plddt, "n_residues": residues}
