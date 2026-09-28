"""Where the models and the data are: search over the public repositories.

:mod:`hallsim.search.models` finds a deposited model (BioModels, JWS
Online, ModelDB, BioSimulations, Physiome) or a paper whose supplement
carries one; :mod:`hallsim.search.datasets` finds a dataset (GEO, Zenodo,
PRIDE, MetaboLights, Metabolomics Workbench, ArrayExpress, the BioImage
Archive, NASA's OSDR) and reads its design from its sample titles;
:mod:`hallsim.search.literature` reads Europe PMC full text for where a
paper says its model lives; :mod:`hallsim.search.web` follows those
pointers across the web. :mod:`hallsim.search.fetch` is the one network
and cache layer under all four.

Nothing here imports the rest of the package, and a test keeps it so. A
hit is a record of what a repository holds; whether the framework can use
it — a model that emits a quantity, a dataset that measures one — is
decided in :mod:`hallsim.screens`.
"""
