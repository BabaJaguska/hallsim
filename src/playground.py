from hallsim.hallmark_registry import (
    create_mitochondrial_dysfunction_hallmark,
    create_disabled_autophagy_hallmark,
    list_hallmarks,
)


def main():
    hallmarks_available = list_hallmarks()
    print("Available hallmarks:", hallmarks_available)
    mito_hallmark = create_mitochondrial_dysfunction_hallmark(handle=0.5)
    autophagy_hallmark = create_disabled_autophagy_hallmark(handle=0.8)

    print("Mitochondrial Dysfunction Hallmark:", mito_hallmark)
    print("Disabled Autophagy Hallmark:", autophagy_hallmark)

    xx = mito_hallmark.get_parameter_values()
    print("Mitochondrial Dysfunction Parameter Values:", xx)
    mito_hallmark = mito_hallmark.set_handle(0.9)
    print("Updated Mitochondrial Dysfunction Hallmark:", mito_hallmark)
    mito_params = mito_hallmark.get_parameter_values()
    print("Updated Mitochondrial Dysfunction Parameter Values:", mito_params)
    mito_hallmark = mito_hallmark.intervene(delta=-0.3)
    print("Intervened Mitochondrial Dysfunction Hallmark:", mito_hallmark)
    mito_params = mito_hallmark.get_parameter_values()
    print(
        "Intervened Mitochondrial Dysfunction Parameter Values:", mito_params
    )


if __name__ == "__main__":
    main()
