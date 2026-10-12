#!/usr/bin/env python3

"""Landau V5: uslovljena energija kvazicestica i njihovih interakcija."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, D, WINDOW, CONTEXT = 39, 7, 8, 64, 17
ALPHAS = (.01, .1, 1.)

angle = (
    2*np.pi*np.arange(1, 40)[:, None]
    * np.arange(1, 5)[None, :]/40.
)
U = np.column_stack((
    np.cos(angle), np.sin(angle)
))/np.sqrt(K)

PAIRS = tuple(
    (i, j) for i in range(D) for j in range(i, D)
)
WIDTH = D*CONTEXT+len(PAIRS)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def occupations(rows):
    rows = np.atleast_2d(rows)
    result = np.zeros((len(rows), N))

    result[
        np.arange(len(rows))[:, None],
        rows-1
    ] = 1.

    return result


def profiles(rows):
    # Projekcija 0/1 zauzeca jedne CELE kombinacije.
    # Ne racunaju se istorijske frekvencije brojeva.
    return occupations(rows) @ U


def structure(row):
    return np.diff(np.r_[0., row, 40.])/5.


def context(history, t):
    return np.r_[
        1.,
        structure(history[t-1]),
        structure(history[t-2])
    ]


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def quadratic(delta):
    return np.column_stack([
        (.5 if i == j else 1.)
        * delta[:, i]*delta[:, j]
        for i, j in PAIRS
    ])


def design(rows, reference, c):
    delta = profiles(rows)-profiles(reference)

    first = (
        delta[:, :, None]*c[None, None, :]
    ).reshape(len(delta), D*CONTEXT)

    return np.column_stack((
        first, quadratic(delta)
    ))


def prepare(history):
    if len(history) < WINDOW+2:
        raise ValueError(
            "Potrebno je najmanje 66 prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    gram = np.zeros((WIDTH, WIDTH))
    rhs = np.zeros(WIDTH)

    for t in range(2, len(history)):
        c = context(history, t)

        actual = design(
            history[t:t+1], history[t-1], c
        )
        negative = design(
            replacements(history[t]),
            history[t-1],
            c
        )

        # Energija stvarnog naslednika treba da bude niza.
        contrasts = negative-actual

        gram += (
            contrasts.T @ contrasts/len(contrasts)
        )
        rhs += contrasts.sum(axis=0)/len(contrasts)

    gram /= WINDOW
    rhs /= WINDOW

    candidates = np.unique(
        np.vstack(
            [history[2:]]
            + [
                replacements(row)
                for row in history[2:]
            ]
        ),
        axis=0
    )

    return (
        gram,
        rhs,
        context(history, len(history)),
        profiles(history[-1])[0],
        candidates
    )


def fit(prepared, alpha):
    gram, rhs, c, reference = prepared[:4]

    coef = np.linalg.solve(
        gram+alpha*np.eye(WIDTH), rhs
    )

    field = (
        coef[:D*CONTEXT].reshape(D, CONTEXT) @ c
    )
    interaction = np.zeros((D, D))

    for value, (i, j) in zip(
        coef[D*CONTEXT:], PAIRS
    ):
        interaction[i, j] = interaction[j, i] = value

    return field, interaction, reference


def energy(rows, model):
    field, interaction, reference = model
    delta = profiles(rows)-reference

    return (
        delta @ field
        + .5*np.einsum(
            "bi,ij,bj->b",
            delta, interaction, delta
        )
    )


def quasiparticle_energies(row, model):
    field, interaction, reference = model
    delta = profiles(row)[0]-reference

    return U @ (field+interaction @ delta)


def choose(prepared, model):
    candidates = prepared[4]
    values = energy(candidates, model)

    if not np.all(np.isfinite(values)):
        raise ArithmeticError(
            "Neispravna energija interakcija."
        )

    best = int(np.argmin(values))

    return (
        tuple(map(int, candidates[best])),
        float(values[best])
    )


def predict(prepared, alpha):
    return choose(prepared, fit(prepared, alpha))


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _ = predict(prepared, alpha)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    alpha = ALPHAS[int(np.argmax(means))]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={max(means):.4f}",
        flush=True
    )

    scores, previous, independent = [], [], []

    # Alpha ostaje fiksna tokom testa.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        prepared = prepare(rows[:t])
        model = fit(prepared, alpha)
        combo, _ = choose(prepared, model)

        scores.append(hits(combo, rows[t]))
        previous.append(hits(rows[t-1], rows[t]))

        # Isti kandidati i isto polje,
        # sa iskljucenom matricom interakcija.
        control = (
            model[0],
            np.zeros((D, D)),
            model[2]
        )
        reference, _ = choose(prepared, control)
        independent.append(hits(reference, rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V5 FERMI-LIQUID", scores)
    summarize("Prethodna kombinacija", previous)
    summarize(
        "Isto polje bez interakcija", independent
    )

    prepared = prepare(rows)
    model = fit(prepared, alpha)
    combo, value = choose(prepared, model)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"E={value:.10f}; "
        f"kandidata={len(prepared[4])}"
    )
    print(
        f"Norma matrice interakcija="
        f"{np.linalg.norm(model[1]):.6f}"
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )
    print(
        "E rangira cele kombinacije; "
        "nije verovatnoca."
    )


def check_swaps(row, model):
    alternatives = replacements(row)
    delta = profiles(alternatives)-profiles(row)

    dressed = quasiparticle_energies(row, model)
    changes = (
        occupations(alternatives)-occupations(row)
    )

    expected = (
        changes @ dressed
        + .5*np.einsum(
            "bi,ij,bj->b",
            delta, model[1], delta
        )
    )
    actual = (
        energy(alternatives, model)
        - energy(row, model)[0]
    )

    assert np.allclose(
        actual, expected,
        atol=1e-9, rtol=1e-9
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (20, 1))
    prepared = prepare(history)

    assert (
        np.linalg.eigvalsh(prepared[0]).min()
        >= -1e-8
    )

    for row in base:
        occ = occupations(row)[0]
        assert np.count_nonzero(occ) == K
        assert occ.sum() == K

        neighbors = replacements(row)
        assert (
            len(neighbors)
            == len(np.unique(neighbors, axis=0))
            == 224
        )
        assert all(
            hits(row, x) == 6 for x in neighbors
        )

    for alpha in ALPHAS:
        model = fit(prepared, alpha)

        assert np.allclose(
            model[1], model[1].T
        )

        result = choose(prepared, model)
        assert result == predict(
            prepare(history), alpha
        )
        assert any(
            np.array_equal(result[0], x)
            for x in prepared[4]
        )

        for row in base:
            check_swaps(row, model)

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(altered[:-1]), .1)
    )

    print(
        "Self-test OK: zauzeca, kvazicestice, "
        "svih 224 zamena i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=1; prosek=1.4375
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V5 FERMI-LIQUID: prosek=1.1562; pogodaka 0..7=[13, 31, 17, 3, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Isto polje bez interakcija: prosek=1.1406; pogodaka 0..7=[13, 33, 14, 4, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 6,7,8,9,15,29,32
E=-0.5376452701; kandidata=14396
Norma matrice interakcija=0.151043
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
E rangira cele kombinacije; nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=0.01; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V5 FERMI-LIQUID: prosek=1.3281; pogodaka 0..7=[13, 25, 18, 8, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Isto polje bez interakcija: prosek=1.3906; pogodaka 0..7=[13, 24, 18, 7, 2, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 4,5,7,16,28,29,30
E=-1.8964331428; kandidata=14392
Norma matrice interakcija=1.227234
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
E rangira cele kombinacije; nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=1; prosek=1.2917
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V5 FERMI-LIQUID: prosek=1.0156; pogodaka 0..7=[19, 27, 16, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Isto polje bez interakcija: prosek=1.2969; pogodaka 0..7=[12, 28, 18, 5, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 3,5,12,13,14,15,23
E=-0.6579717101; kandidata=14388
Norma matrice interakcija=0.135695
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
E rangira cele kombinacije; nije verovatnoca.
"""
 


"""
Uklonjeni su prosečna referentna distribucija i početni aritmetički nizovi.

Fajl	                Sledeća kombinacija	V5 — prosek na 64 kola	Prethodna kombinacija	Bez interakcija
Objedinjeni, red 4699	6,7,8,9,15,29,32	1,1562	                1,2188	                1,1406
Loto, red 2972	        4,5,7,16,28,29,30	1,3281	                1,2969	                1,3906
Loto Plus, red 1728	    3,5,12,13,14,15,23	1,0156	                1,2969	                1,2969

Proverene su efektivne energije kvazičestica, svih 224 pojedinačnih zamena, ponovljivost i hronologija. 
Pretraga obuhvata poslednje 64 kombinacije i sve njihove zamene jednog broja.
"""
