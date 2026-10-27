"""M4 templates and template synthesis — regressions for two bugs found when
the pipeline was first run end to end.
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline.phoneme_templates import split_thirds
from src.baseline.synthesize_baseline import synthesize_from_templates


def _templates(*phones, value=1.0):
    return {p: np.full((80, 3), value, dtype=np.float32) for p in phones}


# ---------------------------------------------------------------------------
# Bug 1: `templates.get(phone) or templates.get(fallback)` raised ValueError
# ---------------------------------------------------------------------------


def test_synthesis_works_for_a_phone_that_has_a_template():
    """Regression: truth-testing the (80, 3) array raised ValueError.

    `a or b` calls bool(a); for an array with >1 element numpy refuses. This
    crashed A1 on its very first phone.
    """
    clip = pd.DataFrame({"phone": ["AA1"], "n_frames": [4]})
    mel = synthesize_from_templates(clip, _templates("AA1", "sil"))
    assert mel.shape == (80, 4)
    assert not np.isnan(mel).any()


def test_synthesis_falls_back_when_a_phone_has_no_template():
    clip = pd.DataFrame({"phone": ["ZZZ"], "n_frames": [3]})
    mel = synthesize_from_templates(clip, _templates("sil", value=0.5))
    assert mel.shape == (80, 3)
    assert np.allclose(mel, 0.5)


def test_synthesis_survives_no_template_at_all():
    clip = pd.DataFrame({"phone": ["ZZZ"], "n_frames": [3]})
    mel = synthesize_from_templates(clip, {})
    assert mel.shape == (80, 3)


@pytest.mark.parametrize("durs", [[1], [2], [3], [1, 2, 3], [5, 1, 8, 2], [7] * 12])
def test_synthesised_mel_is_exactly_the_requested_length(durs):
    clip = pd.DataFrame({"phone": ["AA1"] * len(durs), "n_frames": durs})
    mel = synthesize_from_templates(clip, _templates("AA1"))
    assert mel.shape == (80, sum(durs))


# ---------------------------------------------------------------------------
# Bug 2: empty state slices made the mean NaN, poisoning 61 of 70 templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 10, 70])
def test_split_thirds_stays_inside_the_segment(d):
    p1, p2 = split_thirds(d)
    assert 0 <= p1 <= p2 <= d


def test_build_templates_has_no_nan_for_one_and_two_frame_phones(tmp_path, monkeypatch):
    """Regression: 1- and 2-frame phones left NaN in 61 of 70 templates.

    3% of phones in phones.csv are <= 2 frames, and because a template is the
    mean over occurrences, one NaN occurrence poisons the whole phone.
    """
    import src.baseline.phoneme_templates as pt
    monkeypatch.setattr(pt, "MEL_DIR", tmp_path)

    mel = np.arange(80 * 9, dtype=np.float32).reshape(80, 9)
    np.save(tmp_path / "c.npy", mel)
    phones = pd.DataFrame({
        "clip_id": ["c", "c", "c"],
        "idx": [0, 1, 2],
        "phone": ["AA1", "B", "sil"],
        "start_frame": [0, 1, 3],
        "n_frames": [1, 2, 6],          # the two short ones are the regression
        "word": ["a", "b", "sil"],
        "word_idx": [0, 1, -1],
        "idx_in_word": [0, 0, 0],
        "word_len": [1, 1, 1],
    })

    templates = pt.build_templates(phones, {"c"})
    assert set(templates) == {"AA1", "B", "sil"}
    for phone, arr in templates.items():
        assert arr.shape == (80, 3), phone
        assert not np.isnan(arr).any(), f"{phone} template contains NaN"


def test_a_one_frame_phone_gives_three_identical_states(tmp_path, monkeypatch):
    """With only one frame there is no begin/middle/end, so all three are it."""
    import src.baseline.phoneme_templates as pt
    monkeypatch.setattr(pt, "MEL_DIR", tmp_path)

    mel = np.arange(80 * 4, dtype=np.float32).reshape(80, 4)
    np.save(tmp_path / "c.npy", mel)
    phones = pd.DataFrame({
        "clip_id": ["c"], "idx": [0], "phone": ["AA1"],
        "start_frame": [2], "n_frames": [1],
        "word": ["a"], "word_idx": [0], "idx_in_word": [0], "word_len": [1],
    })

    t = pt.build_templates(phones, {"c"})["AA1"]
    assert np.allclose(t[:, 0], mel[:, 2])
    assert np.allclose(t[:, 1], mel[:, 2])
    assert np.allclose(t[:, 2], mel[:, 2])
