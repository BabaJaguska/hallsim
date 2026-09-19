"""The hallmarks of aging as intents: what each does, in ontology terms,
naming no model.

:func:`hallsim.handles.suggest_registry` turns these into named mappings
for a given composite, which is how a registry such as
``demos.models.hallmarks.HALLMARK_REGISTRY`` is proposed; the proposal is
reviewed and kept as a file, and the file is what gets applied. Ids are
UniProt (proteins), ChEBI (small molecules) and GO (complexes, processes,
compartments). A hallmark whose species a composite does not annotate
suggests nothing, which the suggestion table shows.
"""

from __future__ import annotations

from hallsim.handles import Intent, IntentTarget

LOPEZ_OTIN = "Lopez-Otin et al. 2023"

HALLMARK_INTENTS: dict[str, Intent] = {
    "Genomic Instability": Intent(
        name="Genomic Instability",
        description="More DNA damage is made per unit exposure.",
        category="Primary",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"chebi": "CHEBI:16991"}, "production", 1.0, 2.0),
        ],
    ),
    "Telomere Attrition": Intent(
        name="Telomere Attrition",
        description="A persistent damage response stabilises p53.",
        category="Primary",
        references=[LOPEZ_OTIN, "d'Adda di Fagagna et al. 2003"],
        targets=[
            IntentTarget({"uniprot": "P04637"}, "consumption", 1.0, -0.5),
        ],
    ),
    "Epigenetic Alterations": Intent(
        name="Epigenetic Alterations",
        description="CDK-inhibitor loci are derepressed.",
        category="Primary",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"uniprot": "Q6I9V6"}, "production", 1.0, 1.0),
        ],
    ),
    "Loss of Proteostasis": Intent(
        name="Loss of Proteostasis",
        description="The proteasome degrades less.",
        category="Primary",
        references=[LOPEZ_OTIN, "Proctor et al. 2007"],
        targets=[
            IntentTarget({"go": "GO:0000502"}, "production", 1.0, -1.0),
        ],
    ),
    "Disabled Macroautophagy": Intent(
        name="Disabled Macroautophagy",
        description="Autophagic flux is activated less.",
        category="Primary",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"go": "GO:0000422"}, "production", 1.0, -0.8),
        ],
    ),
    "Deregulated Nutrient Sensing": Intent(
        name="Deregulated Nutrient Sensing",
        description=(
            "mTORC1 is phosphorylated more (+1) or less (-1, rapamycin)."
        ),
        category="Antagonistic",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"go": "GO:0031931"}, "production", 1.0, 0.5),
        ],
    ),
    "Mitochondrial Dysfunction": Intent(
        name="Mitochondrial Dysfunction",
        description="Mitochondria turn dysfunctional faster.",
        category="Antagonistic",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"go": "GO:0005739"}, "production", 1.0, 2.0),
        ],
    ),
    "Cellular Senescence": Intent(
        name="Cellular Senescence",
        description="p21 and SA-beta-gal are made faster.",
        category="Antagonistic",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"uniprot": "P38936"}, "production", 1.0, 1.0),
            IntentTarget({"uniprot": "P16278"}, "production", 1.0, 1.0),
        ],
    ),
    "Stem Cell Exhaustion": Intent(
        name="Stem Cell Exhaustion",
        description="Stem cell maintenance weakens.",
        category="Integrative",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"go": "GO:0019827"}, "production", 1.0, -0.5),
        ],
    ),
    "Altered Intercellular Communication": Intent(
        name="Altered Intercellular Communication",
        description="Endocrine input to Akt falls.",
        category="Integrative",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"uniprot": "P31749"}, "production", 1.0, -0.5),
        ],
    ),
    "Chronic Inflammation": Intent(
        name="Chronic Inflammation",
        description="IKKbeta and JNK are activated more.",
        category="Integrative",
        references=[LOPEZ_OTIN, "Franceschi et al. 2018"],
        targets=[
            IntentTarget({"uniprot": "O14920"}, "production", 1.0, 2.0),
            IntentTarget({"uniprot": "P45983"}, "production", 1.0, 1.0),
        ],
    ),
    "Dysbiosis": Intent(
        name="Dysbiosis",
        description="Microbial products feed IKKbeta activation, weakly.",
        category="Integrative",
        references=[LOPEZ_OTIN],
        targets=[
            IntentTarget({"uniprot": "O14920"}, "production", 1.0, 0.5),
        ],
    ),
}
