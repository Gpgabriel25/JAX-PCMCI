import argparse

from jax_pcmci.precompile import warmup_common_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompile PCMCI kernels for benchmark-like configs.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed precompile progress")
    parser.add_argument("--include-large", action="store_true", help="Also warm larger config (N=20, T=2000, tau_max=5)")
    args = parser.parse_args()

    configs = [
        (10, 1000, 5),
    ]

    if args.include_large:
        configs.append((20, 2000, 5))

    warmup_common_configs(configs=configs, verbose=args.verbose)


if __name__ == "__main__":
    main()
