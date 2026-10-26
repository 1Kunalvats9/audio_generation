"""Build the four filelists  (M1).  Run after prepare_metadata:

    python -m src.data.split

Test is WHOLE ID prefixes (whole source texts), so no source text appears in
both train and test. With one speaker that is the only honest split we can make.
Writes filelists/{train,val,test,test_100}.txt - these are committed to git so
all five members evaluate on exactly the same clips.
"""

import numpy as np
import pandas as pd

from src.config import METADATA_CLEAN, N_TEST_100, N_VAL, SEED, TEST_PREFIXES
from src.utils import prefix_of, write_filelist


def make_splits(df: pd.DataFrame):
    """-> (train, val, test, test_100) lists of clip IDs."""
    rng = np.random.default_rng(SEED)

    prefixes = df["id"].map(prefix_of)
    unknown = set(TEST_PREFIXES) - set(prefixes)
    assert not unknown, f"test prefixes not in the data: {unknown}"

    is_test = prefixes.isin(TEST_PREFIXES)
    test = sorted(df.loc[is_test, "id"])
    remainder = sorted(df.loc[~is_test, "id"])

    assert N_VAL <= len(remainder)
    val = sorted(rng.choice(remainder, size=N_VAL, replace=False).tolist())
    train = sorted(set(remainder) - set(val))

    assert N_TEST_100 <= len(test), f"test has {len(test)} clips, need {N_TEST_100}"
    test_100 = sorted(rng.choice(test, size=N_TEST_100, replace=False).tolist())

    # the splits must be a clean partition
    assert set(train) | set(val) | set(test) == set(df["id"])
    assert not (set(train) & set(val)) and not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert set(test_100) <= set(test)
    # no source text may straddle train and test
    assert not (set(map(prefix_of, train)) & set(TEST_PREFIXES))
    return train, val, test, test_100


def main() -> int:
    df = pd.read_csv(METADATA_CLEAN, dtype={"id": str})
    train, val, test, test_100 = make_splits(df)

    for name, ids in [("train", train), ("val", val),
                      ("test", test), ("test_100", test_100)]:
        path = write_filelist(name, ids)
        hours = df.loc[df["id"].isin(ids), "seconds"].sum() / 3600
        print(f"{name:9s} {len(ids):6d} clips  {hours:6.2f} h  -> {path.name}")

    print(f"\ntest prefixes (held-out source texts): {TEST_PREFIXES}")
    print("commit filelists/ so everyone evaluates on the same clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
