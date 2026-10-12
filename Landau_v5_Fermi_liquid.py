#!/usr/bin/env python3

"""LANDAU V5 — FERMI-LIQUID: conditional interacting-quasiparticle energy."""

import argparse
import csv
from pathlib import Path
import numpy as np



DEFAULT = "/data/loto7_4698_k80.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, D, START, WINDOW = 39, 7, 8, 32, 384

angle = (
    2*np.pi*np.arange(1, N+1)[:, None]
    * np.arange(1, D//2+1)[None, :]/(N+1)
)
U = np.column_stack((np.cos(angle), np.sin(angle)))
PAIRS = tuple(
    (i, j) for i in range(D) for j in range(i, D)
)
CONTEXT = 1+3*D


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(values) != K or values != sorted(values)
                    or len(set(values)) != K
                    or not all(1 <= v <= N for v in values)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def profiles(rows):
    rows = np.atleast_2d(rows)
    return U[rows-1].mean(axis=1)


def context(moments, t):
    reference = moments[max(0, t-32):t].mean(axis=0)
    features = np.r_[
        1.,
        moments[t-1]-reference,
        moments[t-2]-reference,
        moments[max(0, t-8):t].mean(axis=0)-reference,
    ]
    return features, reference


def quadratic_features(delta):
    return np.column_stack([
        (.5 if i == j else 1.)*delta[:, i]*delta[:, j]
        for i, j in PAIRS
    ])


def alternatives(row):
    occupied = set(map(int, row))
    empty = sorted(set(range(1, N+1))-occupied)
    results = set()

    for removed in occupied:
        lower = [v for v in empty if v < removed]
        upper = [v for v in empty if v > removed]
        choices = {empty[0], empty[-1]}

        if lower:
            choices.add(lower[-1])
        if upper:
            choices.add(upper[0])

        for inserted in choices:
            results.add(tuple(sorted(
                (occupied-{removed}) | {inserted}
            )))

    return np.asarray(sorted(results), dtype=int)


def prepare(rows, moments, t):
    blocks = []

    # Sve ciljne kombinacije pri podesavanju prethode redu t.
    for i in range(max(START, t-WINDOW), t):
        ctx, reference = context(moments, i)
        positive = moments[i]-reference
        negative = profiles(alternatives(rows[i]))-reference

        first = (
            (negative-positive)[:, :, None]
            * ctx[None, None, :]
        ).reshape(len(negative), -1)

        second = (
            quadratic_features(negative)
            - quadratic_features(positive[None, :])
        )
        blocks.append(np.column_stack((first, second)))

    design = np.vstack(blocks)
    scale = np.maximum(
        np.sqrt((design*design).mean(axis=0)), 1e-8
    )
    a = design/scale
    ctx, reference = context(moments, t)

    return (
        a.T @ a,
        a.sum(axis=0),
        scale,
        ctx,
        reference,
    )


def fit(prepared, alpha):
    gram, rhs, scale, ctx, reference = prepared
    coefficient = np.linalg.solve(
        gram+alpha*np.eye(len(rhs)), rhs
    )/scale

    field = (
        coefficient[:D*CONTEXT].reshape(D, CONTEXT) @ ctx
    )
    interaction = np.zeros((D, D))

    for value, (i, j) in zip(
        coefficient[D*CONTEXT:], PAIRS
    ):
        interaction[i, j] = value
        interaction[j, i] = value

    return field, interaction, reference


def energy(rows, model):
    field, interaction, reference = model
    delta = profiles(rows)-reference

    return (
        delta @ field
        + .5*np.einsum(
            "bi,ij,bj->b",
            delta,
            interaction,
            delta,
            optimize=True,
        )
    )


def quasiparticle_energies(row, model):
    field, interaction, reference = model
    delta = profiles([row])[0]-reference
    return U @ (field+interaction @ delta)/K


def seeds(history):
    candidates = [
        tuple(map(int, row)) for row in history[-32:]
    ]
    for step in range(1, 7):
        for first in range(1, N-(K-1)*step+1):
            candidates.append(tuple(
                first+j*step for j in range(K)
            ))
    return np.asarray(sorted(set(candidates)), dtype=int)


def search(history, model):
    current = seeds(history)
    values = np.arange(1, N+1)

    for iteration in range(1, 101):
        available = np.all(
            values[None, :, None] != current[:, None, :],
            axis=2,
        )
        who, which = np.nonzero(available)
        blocks = [current]
        owners = [np.arange(len(current))]

        # Sve zamene jednog zauzetog broja jednim slobodnim.
        for j in range(K):
            neighbors = current[who].copy()
            neighbors[:, j] = values[which]
            blocks.append(np.sort(neighbors, axis=1))
            owners.append(who)

        candidates = np.vstack(blocks)
        owners = np.concatenate(owners)
        scores = energy(candidates, model)

        minimum = np.full(len(current), np.inf)
        np.minimum.at(minimum, owners, scores)

        eligible = scores <= minimum[owners]+1e-12
        indices = np.arange(len(candidates))
        chosen = np.full(
            len(current), len(candidates), dtype=int
        )
        np.minimum.at(
            chosen,
            owners,
            np.where(eligible, indices, len(candidates)),
        )

        updated = candidates[chosen]
        if np.array_equal(updated, current):
            scores = energy(current, model)
            best = int(np.argmin(scores))
            return (
                tuple(map(int, current[best])),
                float(scores[best]),
                iteration,
            )
        current = updated

    raise RuntimeError(
        "Pretraga nije dostigla optimum zamena u 100 koraka."
    )


def independent_prediction(model):
    field, _, _ = model
    chosen = np.argsort(
        U @ field, kind="stable"
    )[:K]+1
    return tuple(sorted(map(int, chosen)))


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ]), (50, 1))

    moments = profiles(rows)
    model = fit(prepare(rows, moments, 170), 10.)

    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]
    altered = fit(
        prepare(changed, profiles(changed), 170), 10.
    )
    for a, b in zip(model, altered):
        assert np.array_equal(a, b)

    assert np.allclose(model[1], model[1].T)

    row = rows[169]
    candidate = alternatives(row)[0]
    difference = (
        profiles([candidate])[0]-profiles([row])[0]
    )
    gradient = (
        model[0]+model[1] @ (profiles([row])[0]-model[2])
    )
    expected = (
        gradient @ difference
        + .5*difference @ model[1] @ difference
    )
    assert np.isclose(
        energy([candidate], model)[0]
        - energy([row], model)[0],
        expected,
    )

    old = set(map(int, row))
    new = set(map(int, candidate))
    removed = next(iter(old-new))
    inserted = next(iter(new-old))
    dressed = quasiparticle_energies(row, model)

    assert np.isclose(
        gradient @ difference,
        dressed[inserted-1]-dressed[removed-1],
    )

    result = search(rows[:170], model)
    assert result == search(rows[:170], model)

    combo, score, _ = result
    assert len(set(combo)) == K
    assert list(combo) == sorted(combo)

    neighbors = []
    for removed in combo:
        for inserted in set(range(1, N+1))-set(combo):
            neighbors.append(sorted(
                (set(combo)-{removed}) | {inserted}
            ))

    assert energy(neighbors, model).min() >= score-1e-12

    print(
        "Self-test OK: interakcije, kvazicestice, "
        "hronologija i optimum zamena."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    moments = profiles(rows)
    alphas = (1., 10., 100.)
    validation = [[] for _ in alphas]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows, moments, t)

        for j, alpha in enumerate(alphas):
            model = fit(prepared, alpha)
            combo, _, _ = search(rows[:t], model)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{args.validation}",
                flush=True,
            )

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    alpha = alphas[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    independent_scores = []

    for t in range(split, len(rows)):
        model = fit(
            prepare(rows, moments, t), alpha
        )
        combo, _, _ = search(rows[:t], model)

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))
        independent_scores.append(hits(
            independent_prediction(model), rows[t]
        ))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{args.test}",
                flush=True,
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V5 FERMI-LIQUID", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Isto polje bez interakcija", independent_scores
    )

    model = fit(
        prepare(rows, moments, len(rows)), alpha
    )
    combo, score, iterations = search(rows, model)

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(f"E={score:.10f}; iteracija={iterations}")
    print(
        f"Norma matrice interakcija="
        f"{np.linalg.norm(model[1]):.6f}"
    )
    print(
        "E rangira cele kombinacije. "
        "Pretraga je deterministicka; "
        "globalni minimum nije potvrden."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=1; prosek=1.5417
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V5 FERMI-LIQUID: prosek=1.2969; pogodaka 0..7=[14, 28, 14, 5, 3, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Isto polje bez interakcija: prosek=1.0312; pogodaka 0..7=[20, 25, 16, 3, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 8,x,20,y,31,z,33
E=-1.4735332240; iteracija=11
Norma matrice interakcija=2.486924
E rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=100; prosek=1.3750
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V5 FERMI-LIQUID: prosek=1.1250; pogodaka 0..7=[18, 27, 13, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Isto polje bez interakcija: prosek=1.1094; pogodaka 0..7=[18, 27, 15, 2, 2, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 8,x,15,y,31,z,33
E=-1.4279842420; iteracija=11
Norma matrice interakcija=2.222006
E rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=100; prosek=1.1667
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V5 FERMI-LIQUID: prosek=1.3125; pogodaka 0..7=[16, 21, 19, 7, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Isto polje bez interakcija: prosek=1.1719; pogodaka 0..7=[18, 27, 11, 6, 2, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 6,x,16,y,24,z,33
E=-1.4336529176; iteracija=13
Norma matrice interakcija=2.186660
E rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.
"""

 

"""
V5 LANDAU      FERMI-LIQUID THEORY interacting fermions at low temperatures

LANDAU FERMI-LIQUID. Ideju kvazičestica i njihovih međusobnih uticaja prenosim na predikciju cele kombinacije.

V5 će ocenjivati kombinaciju energetskim funkcionalom sa učenim međusobnim interakcijama. Pri pretrazi, zamena jednog broja menjaće i efektivne ocene ostalih — to je konkretna primena Landauove ideje kvazičestica.

Model uči energetski funkcional cele kombinacije i međusobne interakcije. Efektivna ocena jednog broja zavisi od ostalih izabranih brojeva — primena Landauove ideje kvazičestica. 

Predlog za red 4699:
8, x, 20, y, 31, z, 33

Na poslednja 64 kola:
Metod	Prosečan broj pogodaka
V5	1,2969
Prethodna kombinacija	1,2188
Isto polje bez interakcija	1,0312

Provereni su interakcije, efektivne energije, hronologija, ponovljivost i svih 224 pojedinačnih zamena završne kombinacije. Globalni minimum nije potvrđen.
"""
