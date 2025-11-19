from hallsim.hallmarks import Hallmark, hallmark_factory


def create_mitochondrial_dysfunction_hallmark(handle: float = 0.0) -> Hallmark:
    """
    Mitochondrial Dysfunction hallmark.

    Handle semantics:
        0.0 = healthy mitochondria, optimal function
        1.0 = severe dysfunction, high damage, elevated ROS

    Maps to ERiQ parameters:
        - MITO_DMG_RATE_SA: damage accumulation rate
        - MDR: mitochondrial damage repair rate (inversely)
        - ROS_SA: ROS production scaling

    Associated state variables:
        - mito_damage: structural damage to mitochondria
        - mito_function: functional capacity
        - ROS: reactive oxygen species levels
        - ATP_mito: mitochondrial ATP production
    """
    return hallmark_factory(
        name="Mitochondrial Dysfunction",
        handle=handle,
        description=(
            "Impairment in mitochondrial function leading to reduced ATP production, "
            "increased ROS generation, and accumulation of mitochondrial damage."
        ),
        parameter_mappings={
            # Damage accumulation rate increases with dysfunction
            "MITO_DMG_RATE_SA": lambda h: 1.0 + h * 2.0,  # 1.0→3.0
            # Repair rate decreases with dysfunction
            "MDR": lambda h: 1.0 * (1.0 - 0.7 * h),  # 1.0→0.3
            # ROS production increases with dysfunction
            "ROS_SA": lambda h: 1.0 + h * 1.5,  # 1.0→2.5
        },
        state_associations={
            "mito_damage",
            "mito_function",
            "ROS",
            "ATP_mito",
        },
        category="Primary",
        references=["López-Otín et al. 2023", "ERiQ model"],
    )


def create_disabled_autophagy_hallmark(handle: float = 0.0) -> Hallmark:
    """
    Loss of Autophagy hallmark.

    Handle semantics:
        0.0 = efficient autophagy
        1.0 = severely impaired autophagy

    Maps to ERiQ parameters:
        - AUTOPHAGY_SA: autophagy flux scaling
        - FOXO_SA: FOXO activity (promotes autophagy genes)

    Associated state variables:
        - AUTOPHAGY: autophagy activity level
        - FOXO: transcription factor promoting proteostasis
    """
    return hallmark_factory(
        name="Disabled Autophagy",
        handle=handle,
        description=(
            "Decline in autophagy efficiency and quality control mechanisms. "
            "Leads to accumulation of damaged organelles."
        ),
        parameter_mappings={
            # Autophagy efficiency decreases with impairment
            "AUTOPHAGY_SA": lambda h: 1.0 * (1.0 - 0.8 * h),  # 1.0→0.2
            # FOXO activity (promotes autophagy) decreases
            "FOXO_SA": lambda h: 1.0 * (1.0 - 0.6 * h),  # 1.0→0.4
        },
        state_associations={
            "AUTOPHAGY",
            "FOXO",
            "mTOR",  # mTOR inhibits autophagy
        },
        category="Primary",
        references=["López-Otín et al. 2023", "ERiQ model"],
    )


def create_nutrient_sensing_dysregulation_hallmark(
    handle: float = 0.0,
) -> Hallmark:
    """
    Deregulated Nutrient Sensing hallmark.

    Handle semantics:
        0.0 = balanced mTOR/AMPK/SIRT signaling
        1.0 = dysregulated, chronic mTOR activation, impaired AMPK/SIRT

    Maps to ERiQ parameters:
        - MTOR_SA: mTOR activity scaling (increases with dysregulation)
        - AMPK_SA: AMPK activity scaling (decreases with dysregulation)
        - SIRT_SA: SIRT activity scaling (decreases with dysregulation)
        - NAD_RATE_SA: NAD+ regeneration (decreases with dysregulation)

    Associated state variables:
        - mTOR: mechanistic target of rapamycin
        - AMPK: AMP-activated protein kinase
        - SIRT: sirtuin (NAD+-dependent deacetylase)
        - NAD_ratio: NAD+/NADH ratio
    """
    return hallmark_factory(
        name="Deregulated Nutrient Sensing",
        handle=handle,
        description=(
            "Imbalance in nutrient-sensing pathways (mTOR, AMPK, sirtuins). "
            "Chronic mTOR activation, impaired AMPK response, declining NAD+."
        ),
        parameter_mappings={
            # mTOR activity increases with dysregulation (less inhibition)
            "MTOR_SA": lambda h: 1.0 + h * 0.8,  # 1.0→1.8
            # AMPK responsiveness decreases
            "AMPK_SA": lambda h: 1.0 * (1.0 - 0.5 * h),  # 1.0→0.5
            # SIRT activity decreases (NAD+ dependent)
            "SIRT_SA": lambda h: 1.0 * (1.0 - 0.6 * h),  # 1.0→0.4
            # NAD+ regeneration rate decreases
            "NAD_RATE_SA": lambda h: 1.0 * (1.0 - 0.5 * h),  # 1.0→0.5
        },
        state_associations={
            "mTOR",
            "AMPK",
            "SIRT",
            "NAD_ratio",
            "mTOR_activity",
            "mTOR_integrator_c",
            "PGC1a",
        },
        category="Primary",
        references=["López-Otín et al. 2023", "ERiQ model"],
    )


def create_cellular_senescence_hallmark(handle: float = 0.0) -> Hallmark:
    """
    Factory function to create a genomic instability hallmark.

    This hallmark increases DNA damage production rate in the damage_repair model.

    Args:
        handle: Severity level [0, 1]. 0 = healthy, 1 = severe instability

    Returns:
        Hallmark instance configured for genomic instability
    """
    return hallmark_factory(
        name="Genomic Instability",
        handle=handle,
        description="DNA damage accumulation due to genomic instability",
        parameter_mappings={
            # Increase damage production rate (alpha) with hallmark severity
            "eta_damage_rate": lambda h: 0.5
            + h * 2.0,  # ranges from 0.5 to 2.5
        },
        state_associations={"damage_D"},
        # eta comes from saturating_removal model which is sorta generic damage removal
    )


# Registry of all defined hallmarks
HALLMARK_REGISTRY = {
    "Mitochondrial Dysfunction": create_mitochondrial_dysfunction_hallmark,
    "Disabled Autophagy": create_disabled_autophagy_hallmark,
    "Deregulated Nutrient Sensing": create_nutrient_sensing_dysregulation_hallmark,
    "Genomic Instability": create_cellular_senescence_hallmark,
}


def get_hallmark(name: str, handle: float = 0.0) -> Hallmark:
    """
    Factory function to create a hallmark instance by name from the registry.

    Args:
        name: Hallmark identifier (key in HALLMARK_REGISTRY)
        handle: Initial handle value [0, 1]

    Returns:
        Hallmark instance

    Raises:
        KeyError: if hallmark name not found
    """
    if name not in HALLMARK_REGISTRY:
        available = ", ".join(HALLMARK_REGISTRY.keys())
        raise KeyError(f"Unknown hallmark '{name}'. Available: {available}")
    return HALLMARK_REGISTRY[name](handle=handle)


def list_hallmarks() -> list:
    """Return list of available hallmark identifiers."""
    return list(HALLMARK_REGISTRY.keys())


# Example usage:
# mito_dysfunction = get_hallmark("Mitochondrial Dysfunction", handle=0.3)
