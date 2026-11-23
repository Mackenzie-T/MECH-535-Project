"""
Print the trailing-edge radial distributions of Cz, Cr, beta and density for
Part 1 (compressible + losses) and Part 2 (incompressible, lossless).
"""

from thermodynamics import run_ree_solver


def clean_zero(val):
    """Return 0.0 for floating-point noise so the table doesn't show -0.000."""
    return 0.0 if abs(val) < 1e-9 else val


def print_te_table(title, params, fields):
    """Print one table of trailing-edge values, one row per radial node."""
    i_te = params["i_TE"]
    r = params["RADIUS"]

    print(f"\n{title} (Trailing Edge, i = {i_te})")
    print(
        f"{'r [m]':<10} {'Cz [m/s]':<11} {'Cr [m/s]':<11} {'beta [deg]':<12} {'rho [kg/m3]':<12}"
    )
    print("-" * 58)

    for j in range(len(r)):
        cz = clean_zero(fields["CZ"][i_te, j])
        cr = clean_zero(fields["CR"][i_te, j])
        beta = clean_zero(fields["BETA"][i_te, j])
        rho = clean_zero(fields["DENSITY"][i_te, j])

        print(f"{r[j]:<10.4f} {cz:<11.3f} {cr:<11.3f} {beta:<12.3f} {rho:<12.4f}")


def get_te_tables():
    """Run both cases and print their trailing-edge tables."""
    # Part 1: compressible flow with losses
    params, fields_comp = run_ree_solver(compressible=True, with_losses=True)
    print_te_table("Part 1: Compressible + Losses", params, fields_comp)

    # Part 2: incompressible flow with zero losses
    _, fields_incomp = run_ree_solver(compressible=False, with_losses=False)
    print_te_table("Part 2: Incompressible Lossless", params, fields_incomp)


if __name__ == "__main__":
    get_te_tables()
