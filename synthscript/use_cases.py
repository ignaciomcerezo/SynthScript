from synthscript.loading.external_interfaces.external_interface import ExternalInterface
from synthscript.shared.path_bundle import PathBundle


def setup(
    external_interfaces: list[ExternalInterface],
    paths: PathBundle | None = None,
):
    """
    Downloads all files needed to instanciate the dataset given some external interfaces and a path to store them.
    """
    paths = PathBundle() if paths is None else paths
    parts = set()
    for i, ext_int_1 in enumerate(external_interfaces):
        pm_1 = ext_int_1.parts_managed()
        if pm_1.intersection(parts):
            for ext_int_2 in external_interfaces[:i]:
                pm_2 = ext_int_2.parts_managed()
                if pm_2.intersection(pm_1):
                    break
            raise ValueError(
                f"External interface conflict: {ext_int_2} and "
                f"{ext_int_1} manages the same parts: {pm_1.intersection(pm_2)}"
            )
        pr = ext_int_1.parts_required()
        if not pr.issubset(parts):
            prev_msg = (
                f"(after {external_interfaces[:i]})"
                if i != 0
                else "(it is the first one)"
            )
            raise ValueError(
                f"External interface {ext_int_1} needs parts {pr} to setup, but "
                f"only {parts} is setup when it is called {prev_msg}. Try reordering the external interfaces to solve this, "
                "or perhaps the combination you chose is just incompatible."
            )
        parts.update(pm_1)

    for ext_int_1 in external_interfaces:
        ext_int_1.setup(paths)
