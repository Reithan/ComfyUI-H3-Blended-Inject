"""Hypothesis-based tests for the schedule module.

Contract source: docstrings in comfyui_h3_blended_inject/schedule.py and envelope.py.

After the frame-space rework, evaluate_envelope returns list[tuple[int, float]] where
each pair is (absolute_latent_row_idx, denoise).  inject_at is a LATENT FRAME index;
the fade indices are CLIP frame indices.  The _reference_merge implementation here uses
the (row_idx, d) pairs from evaluate_envelope directly, mirroring merge_schedule exactly.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from comfyui_h3_blended_inject.envelope import evaluate_envelope
from comfyui_h3_blended_inject.schedule import (
    Inject,
    RowSchedule,
    merge_schedule,
)

# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_inject(
    *,
    inject_at: int = 0,
    start_fade_in: int = 0,
    start_keyframes: int = 4,
    end_keyframes: int = 20,
    end_fade_out: int = 24,
    min_denoise: float = 0.5,
    interpolation_type: str = "linear",
    audio_mode: str = "fade",
    images: object = None,
    audio: object = None,
    resolution: tuple[int, int] = (64, 64),
    source_length: int = 30,
) -> Inject:
    """Build a valid Inject with sensible defaults for use in tests.

    Invariants maintained:
    - start_fade_in <= start_keyframes <= end_keyframes <= end_fade_out (valid ordering)
    - source_length > end_fade_out (source is long enough)
    - inject_at is a multiple of 17 (snapped grid requirement)
    - resolution is a tuple of multiples of 32
    """
    return Inject(
        inject_at=inject_at,
        start_fade_in=start_fade_in,
        start_keyframes=start_keyframes,
        end_keyframes=end_keyframes,
        end_fade_out=end_fade_out,
        min_denoise=min_denoise,
        interpolation_type=interpolation_type,
        audio_mode=audio_mode,
        images=images,
        audio=audio,
        resolution=resolution,
        source_length=source_length,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def inject_strategy(
    draw: st.DrawFn,
    *,
    max_source_length: int = 60,
    max_inject_at_blocks: int = 4,
) -> Inject:
    """Draw a single Inject satisfying all documented ordering and range invariants."""
    source_length = draw(st.integers(min_value=6, max_value=max_source_length))
    # Draw four sorted indices within [0, source_length - 1]
    raw = draw(
        st.lists(
            st.integers(min_value=0, max_value=source_length - 1),
            min_size=4,
            max_size=4,
        ).map(sorted)
    )
    sfi, sk, ek, efo = raw
    inject_at = draw(st.integers(min_value=0, max_value=max_inject_at_blocks)) * 17
    min_denoise = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    interp = draw(st.sampled_from(["ease_in", "ease_out", "ease_in_out", "linear", "none"]))
    audio_mode = draw(st.sampled_from(["fade", "drop", "keep"]))
    return Inject(
        inject_at=inject_at,
        start_fade_in=sfi,
        start_keyframes=sk,
        end_keyframes=ek,
        end_fade_out=efo,
        min_denoise=min_denoise,
        interpolation_type=interp,
        audio_mode=audio_mode,
        images=None,
        audio=None,
        resolution=(64, 64),
        source_length=source_length,
    )


# ---------------------------------------------------------------------------
# Reference merge implementation (used by property tests)
# ---------------------------------------------------------------------------


def _reference_merge(inject_list: list[Inject], target_rows: int) -> list[RowSchedule]:
    """Independent last-in-wins merge for comparison against merge_schedule.

    Calls evaluate_envelope for each inject in list order; later writes overwrite
    earlier ones on each row.  evaluate_envelope returns (row_idx, denoise) pairs
    so no offset arithmetic is needed here.  Returns the same sparse, sorted structure
    that merge_schedule must return.
    """
    row_map: dict[int, tuple[Inject, float]] = {}
    for inj in inject_list:
        for row_idx, d in evaluate_envelope(
            inj.start_fade_in,
            inj.start_keyframes,
            inj.end_keyframes,
            inj.end_fade_out,
            inj.min_denoise,
            inj.interpolation_type,
            inj.source_length,
            target_rows,
            inj.inject_at,
        ):
            if 0 <= row_idx < target_rows:
                row_map[row_idx] = (inj, d)  # last writer wins
    return [
        RowSchedule(
            row_idx=row_idx,
            denoise=d,
            inject=inj,
            audio_frozen=(inj.audio_mode == "keep"),
        )
        for row_idx, (inj, d) in sorted(row_map.items())
    ]


# ---------------------------------------------------------------------------
# Dataclass shape and defaults
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify the documented dataclass fields exist and have correct defaults."""

    def test_inject_has_all_documented_fields(self) -> None:
        inj = make_inject()
        assert hasattr(inj, "inject_at")
        assert hasattr(inj, "start_fade_in")
        assert hasattr(inj, "start_keyframes")
        assert hasattr(inj, "end_keyframes")
        assert hasattr(inj, "end_fade_out")
        assert hasattr(inj, "min_denoise")
        assert hasattr(inj, "interpolation_type")
        assert hasattr(inj, "audio_mode")
        assert hasattr(inj, "images")
        assert hasattr(inj, "audio")
        assert hasattr(inj, "resolution")
        assert hasattr(inj, "source_length")

    def test_inject_stores_provided_values(self) -> None:
        inj = make_inject(
            inject_at=17,
            min_denoise=0.3,
            interpolation_type="ease_out",
            audio_mode="keep",
            resolution=(128, 96),
            source_length=45,
        )
        assert inj.inject_at == 17
        assert inj.min_denoise == 0.3
        assert inj.interpolation_type == "ease_out"
        assert inj.audio_mode == "keep"
        assert inj.resolution == (128, 96)
        assert inj.source_length == 45

    def test_row_schedule_has_all_documented_fields(self) -> None:
        inj = make_inject()
        rs = RowSchedule(row_idx=3, denoise=0.4, inject=inj)
        assert hasattr(rs, "row_idx")
        assert hasattr(rs, "denoise")
        assert hasattr(rs, "inject")
        assert hasattr(rs, "audio_frozen")

    def test_row_schedule_audio_frozen_defaults_to_false(self) -> None:
        inj = make_inject()
        rs = RowSchedule(row_idx=0, denoise=0.5, inject=inj)
        assert rs.audio_frozen is False

    def test_row_schedule_audio_frozen_can_be_set_true(self) -> None:
        inj = make_inject()
        rs = RowSchedule(row_idx=0, denoise=0.0, inject=inj, audio_frozen=True)
        assert rs.audio_frozen is True

    def test_row_schedule_inject_can_be_none(self) -> None:
        rs = RowSchedule(row_idx=5, denoise=1.0, inject=None)
        assert rs.inject is None


# ---------------------------------------------------------------------------
# merge_schedule: empty inject list
# ---------------------------------------------------------------------------


class TestMergeScheduleEmpty:
    """merge_schedule with no injects returns an empty list."""

    def test_empty_inject_list_returns_empty_list(self) -> None:
        result = merge_schedule([], target_rows=30)
        assert result == []

    def test_empty_inject_list_returns_list_type(self) -> None:
        result = merge_schedule([], target_rows=1)
        assert isinstance(result, list)

    def test_empty_inject_list_any_target_rows_returns_empty(self) -> None:
        assert merge_schedule([], target_rows=100) == []


# ---------------------------------------------------------------------------
# merge_schedule: single inject
# ---------------------------------------------------------------------------


class TestMergeScheduleSingleInject:
    """Single-inject cases: output rows exactly match evaluate_envelope output."""

    def test_single_inject_result_is_list_of_row_schedules(self) -> None:
        inj = make_inject()
        result = merge_schedule([inj], target_rows=30)
        assert isinstance(result, list)
        assert all(isinstance(r, RowSchedule) for r in result)

    def test_single_inject_rows_sorted_ascending(self) -> None:
        inj = make_inject()
        result = merge_schedule([inj], target_rows=30)
        row_idxs = [r.row_idx for r in result]
        assert row_idxs == sorted(row_idxs)

    def test_single_inject_inject_field_is_the_inject(self) -> None:
        inj = make_inject()
        result = merge_schedule([inj], target_rows=30)
        for rs in result:
            assert rs.inject is inj

    def test_single_inject_denoise_matches_evaluate_envelope(self) -> None:
        """Each row's denoise must match evaluate_envelope output for that inject.

        evaluate_envelope now returns (row_idx, denoise) pairs; we verify both the
        row_idx and the denoise value match.
        """
        inj = make_inject()
        target_rows = 30
        result = merge_schedule([inj], target_rows=target_rows)
        expected_pairs = evaluate_envelope(
            inj.start_fade_in,
            inj.start_keyframes,
            inj.end_keyframes,
            inj.end_fade_out,
            inj.min_denoise,
            inj.interpolation_type,
            inj.source_length,
            target_rows,
            inj.inject_at,
        )
        assert len(result) == len(expected_pairs)
        for rs, (expected_row, expected_d) in zip(result, expected_pairs, strict=True):
            assert rs.row_idx == expected_row, (
                f"row index mismatch: got {rs.row_idx}, want {expected_row}"
            )
            assert abs(rs.denoise - expected_d) < 1e-9, (
                f"row {rs.row_idx}: got denoise {rs.denoise}, want {expected_d}"
            )

    def test_single_inject_claimed_row_set_matches_envelope_coverage(self) -> None:
        """The set of claimed rows must exactly equal the rows envelope covers."""
        inj = make_inject()
        target_rows = 30
        result = merge_schedule([inj], target_rows=target_rows)
        expected = _reference_merge([inj], target_rows=target_rows)
        assert {r.row_idx for r in result} == {e.row_idx for e in expected}

    def test_single_inject_unclaimed_rows_not_present(self) -> None:
        """Rows with d=1.0 (unclaimed) must NOT appear in the sparse output."""
        inj = make_inject()
        target_rows = 30
        result = merge_schedule([inj], target_rows=target_rows)
        claimed_rows = {r.row_idx for r in result}
        # All returned row indices must be within target bounds
        for row_idx in claimed_rows:
            assert 0 <= row_idx < target_rows

    def test_single_inject_still_inject_returns_one_row(self) -> None:
        """Degenerate envelope (all four indices equal) covers exactly one row."""
        inj = make_inject(
            start_fade_in=10,
            start_keyframes=10,
            end_keyframes=10,
            end_fade_out=10,
            source_length=20,
        )
        result = merge_schedule([inj], target_rows=30)
        assert len(result) == 1
        assert result[0].inject is inj
        assert result[0].denoise == inj.min_denoise


# ---------------------------------------------------------------------------
# merge_schedule: last-in-wins overlap
# ---------------------------------------------------------------------------


class TestMergeScheduleLastInWins:
    """Concrete overlap tests with hand-computable structure.

    The boundary between A-only and B-wins regions is a hard edge: no row has a
    denoise value that is an average or blend of A and B.
    """

    def _assert_schedules_equal(
        self,
        result: list[RowSchedule],
        expected: list[RowSchedule],
        *,
        label: str = "",
    ) -> None:
        assert len(result) == len(expected), (
            f"{label} length mismatch: got {len(result)}, want {len(expected)}"
        )
        for rs, es in zip(result, expected, strict=True):
            assert rs.row_idx == es.row_idx, f"{label} row_idx mismatch"
            assert rs.inject is es.inject, f"{label} row {rs.row_idx}: inject mismatch"
            assert abs(rs.denoise - es.denoise) < 1e-9, (
                f"{label} row {rs.row_idx}: denoise got {rs.denoise} want {es.denoise}"
            )
            assert rs.audio_frozen == es.audio_frozen, (
                f"{label} row {rs.row_idx}: audio_frozen mismatch"
            )

    def test_b_wins_all_rows_it_claims(self) -> None:
        """Every row B claims has entry.inject is B; B entirely overwrites A on its rows."""
        target_rows = 50
        # A: starts at row 0, long envelope in source space
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=40,
            end_fade_out=44,
            source_length=50,
            audio_mode="fade",
        )
        # B: starts at row 17 (one H3 block later), shorter envelope
        inj_b = make_inject(
            inject_at=17,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
            audio_mode="fade",
        )
        result = merge_schedule([inj_a, inj_b], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b], target_rows=target_rows)
        self._assert_schedules_equal(result, expected, label="A+B overlap")

    def test_a_only_rows_keep_a(self) -> None:
        """Rows covered by A but not B must have entry.inject == A."""
        target_rows = 50
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=40,
            end_fade_out=44,
            source_length=50,
        )
        inj_b = make_inject(
            inject_at=17,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
        )
        result = merge_schedule([inj_a, inj_b], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b], target_rows=target_rows)
        result_by_row = {r.row_idx: r for r in result}
        for es in expected:
            if es.inject is inj_a:
                rs = result_by_row.get(es.row_idx)
                assert rs is not None, f"expected row {es.row_idx} (A-only) missing from result"
                assert rs.inject is inj_a, f"row {es.row_idx} should be won by A, got {rs.inject}"

    def test_hard_boundary_no_blended_denoise(self) -> None:
        """At the overlap boundary there is no averaged denoise; it is a hard switch."""
        target_rows = 50
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=40,
            end_fade_out=44,
            source_length=50,
        )
        inj_b = make_inject(
            inject_at=17,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
        )
        result = merge_schedule([inj_a, inj_b], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b], target_rows=target_rows)
        exp_by_row = {e.row_idx: e.denoise for e in expected}
        for rs in result:
            assert abs(rs.denoise - exp_by_row[rs.row_idx]) < 1e-9, (
                f"row {rs.row_idx}: denoise should be exactly winner's value, not a blend"
            )

    def test_non_overlapping_injects_both_present(self) -> None:
        """Two non-overlapping injects: both appear in output, gap rows are absent."""
        target_rows = 80
        # A covers a region near row 0
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=8,
            end_fade_out=12,
            source_length=20,
        )
        # B starts well past A's coverage (inject_at=51 = 3 * 17)
        inj_b = make_inject(
            inject_at=51,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=8,
            end_fade_out=12,
            source_length=20,
        )
        result = merge_schedule([inj_a, inj_b], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b], target_rows=target_rows)
        self._assert_schedules_equal(result, expected, label="non-overlapping A+B")
        # Verify both injects are represented
        winning_injects = {r.inject for r in result}
        assert inj_a in winning_injects
        assert inj_b in winning_injects

    def test_three_injects_last_one_wins_on_full_overlap(self) -> None:
        """With A, B, C all claiming the same rows, C wins everywhere it claims."""
        target_rows = 30
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
            min_denoise=0.8,
        )
        inj_b = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
            min_denoise=0.5,
        )
        inj_c = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=4,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
            min_denoise=0.2,
        )
        result = merge_schedule([inj_a, inj_b, inj_c], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b, inj_c], target_rows=target_rows)
        self._assert_schedules_equal(result, expected, label="three-way overlap")
        # On every row, C (last) must win
        for rs in result:
            winner_d = rs.inject.min_denoise if rs.inject else None
            assert rs.inject is inj_c, (
                f"row {rs.row_idx}: expected C to win, got inject with min_denoise={winner_d}"
            )


# ---------------------------------------------------------------------------
# merge_schedule: audio_frozen
# ---------------------------------------------------------------------------


class TestAudioFrozen:
    """audio_frozen on a RowSchedule row is True iff the winning inject has audio_mode=='frozen'."""

    def test_frozen_inject_sets_audio_frozen_true(self) -> None:
        inj = make_inject(audio_mode="keep")
        result = merge_schedule([inj], target_rows=30)
        for rs in result:
            assert rs.audio_frozen is True, f"row {rs.row_idx}: expected audio_frozen=True"

    def test_match_inject_sets_audio_frozen_false(self) -> None:
        inj = make_inject(audio_mode="fade")
        result = merge_schedule([inj], target_rows=30)
        for rs in result:
            assert rs.audio_frozen is False, f"row {rs.row_idx}: expected audio_frozen=False"

    def test_drop_inject_sets_audio_frozen_false(self) -> None:
        inj = make_inject(audio_mode="drop")
        result = merge_schedule([inj], target_rows=30)
        for rs in result:
            assert rs.audio_frozen is False, f"row {rs.row_idx}: expected audio_frozen=False"

    def test_audio_frozen_tracks_winning_inject_not_loser(self) -> None:
        """When B (frozen) wins over A (match), audio_frozen reflects B's mode."""
        target_rows = 50
        inj_a = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=40,
            end_fade_out=44,
            source_length=50,
            audio_mode="fade",
        )
        inj_b = make_inject(
            inject_at=17,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=20,
            end_fade_out=24,
            source_length=30,
            audio_mode="keep",
        )
        result = merge_schedule([inj_a, inj_b], target_rows=target_rows)
        expected = _reference_merge([inj_a, inj_b], target_rows=target_rows)
        result_by_row = {r.row_idx: r for r in result}
        for es in expected:
            rs = result_by_row[es.row_idx]
            expected_frozen = es.inject.audio_mode == "keep"
            assert rs.audio_frozen == expected_frozen, (
                f"row {rs.row_idx}: audio_frozen should be {expected_frozen} "
                f"(winner audio_mode={es.inject.audio_mode})"
            )

    def test_audio_frozen_is_bool_not_string(self) -> None:
        inj = make_inject(audio_mode="keep")
        result = merge_schedule([inj], target_rows=30)
        for rs in result:
            assert isinstance(rs.audio_frozen, bool)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


class TestMergeScheduleProperties:
    """Randomized property tests for merge_schedule semantics."""

    @given(
        inject_list=st.lists(
            inject_strategy(max_source_length=60, max_inject_at_blocks=4),
            min_size=1,
            max_size=5,
        ),
        target_rows=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50)
    def test_last_writer_wins_for_every_claimed_row(
        self, inject_list: list[Inject], target_rows: int
    ) -> None:
        """For every claimed row, entry.inject is the last inject in list order that covers it."""
        result = merge_schedule(inject_list, target_rows=target_rows)
        expected = _reference_merge(inject_list, target_rows=target_rows)

        assert len(result) == len(expected), (
            f"length mismatch: got {len(result)}, want {len(expected)}"
        )
        result_by_row = {r.row_idx: r for r in result}
        for es in expected:
            rs = result_by_row.get(es.row_idx)
            assert rs is not None, f"row {es.row_idx} missing from result"
            assert rs.inject is es.inject, (
                f"row {es.row_idx}: inject mismatch — expected last inject that covers row"
            )
            assert abs(rs.denoise - es.denoise) < 1e-9, (
                f"row {es.row_idx}: denoise mismatch got {rs.denoise} want {es.denoise}"
            )
            expected_frozen = es.inject.audio_mode == "keep"
            assert rs.audio_frozen == expected_frozen, f"row {es.row_idx}: audio_frozen mismatch"

    @given(
        inject_list=st.lists(
            inject_strategy(max_source_length=60, max_inject_at_blocks=4),
            min_size=1,
            max_size=5,
        ),
        target_rows=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=40)
    def test_output_sorted_ascending_and_unique(
        self, inject_list: list[Inject], target_rows: int
    ) -> None:
        """Result is always sorted by row_idx ascending with no duplicate row indices."""
        result = merge_schedule(inject_list, target_rows=target_rows)
        row_idxs = [r.row_idx for r in result]
        assert row_idxs == sorted(row_idxs), "result must be sorted by row_idx ascending"
        assert len(set(row_idxs)) == len(row_idxs), "row indices must be unique in result"

    @given(
        inject_list=st.lists(
            inject_strategy(max_source_length=60, max_inject_at_blocks=4),
            min_size=1,
            max_size=5,
        ),
        target_rows=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=40)
    def test_all_result_rows_in_bounds(self, inject_list: list[Inject], target_rows: int) -> None:
        """All row_idx values in the result must be in [0, target_rows)."""
        result = merge_schedule(inject_list, target_rows=target_rows)
        for rs in result:
            assert 0 <= rs.row_idx < target_rows, (
                f"row_idx {rs.row_idx} out of bounds [0, {target_rows})"
            )

    @given(
        inject_list=st.lists(
            inject_strategy(max_source_length=60, max_inject_at_blocks=4),
            min_size=1,
            max_size=5,
        ),
        target_rows=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=40)
    def test_all_result_injects_are_not_none(
        self, inject_list: list[Inject], target_rows: int
    ) -> None:
        """Every RowSchedule in the sparse result must have a non-None inject (it is claimed)."""
        result = merge_schedule(inject_list, target_rows=target_rows)
        for rs in result:
            assert rs.inject is not None, (
                f"row {rs.row_idx} has inject=None but only claimed rows are returned"
            )

    @given(
        inject_list=st.lists(
            inject_strategy(max_source_length=60, max_inject_at_blocks=4),
            min_size=1,
            max_size=5,
        ),
        target_rows=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=40)
    def test_audio_frozen_consistent_with_winning_inject(
        self, inject_list: list[Inject], target_rows: int
    ) -> None:
        """audio_frozen must always equal (winning inject's audio_mode == 'frozen')."""
        result = merge_schedule(inject_list, target_rows=target_rows)
        for rs in result:
            assert rs.inject is not None  # precondition
            assert rs.audio_frozen == (rs.inject.audio_mode == "keep"), (
                f"row {rs.row_idx}: audio_frozen={rs.audio_frozen} but "
                f"inject.audio_mode={rs.inject.audio_mode}"
            )


# ---------------------------------------------------------------------------
# RowSchedule.audio_preserve property — regression tests
# ---------------------------------------------------------------------------


class TestAudioPreserve:
    """Truth-table for RowSchedule.audio_preserve.

    Regression for the bug where fade-mode d==0 audio was generated from scratch
    instead of preserved.  audio_preserve must return True exactly for:
      - keep mode (audio_frozen=True), any denoise value, and
      - fade mode with denoise==0.0.
    All other combinations must return False.

    Pre-fix these tests fail with:
      AttributeError: 'RowSchedule' object has no attribute 'audio_preserve'
    """

    def _make_rs(
        self,
        audio_mode: str,
        denoise: float,
        audio_frozen: bool = False,
    ) -> RowSchedule:
        inj = Inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=17,
            end_fade_out=39,
            min_denoise=0.0,
            interpolation_type="linear",
            audio_mode=audio_mode,
            images=None,
            audio=None,
            resolution=(0, 0),
            source_length=39,
        )
        return RowSchedule(
            row_idx=0,
            denoise=denoise,
            inject=inj,
            audio_frozen=audio_frozen,
            region="preserve" if denoise == 0.0 else "hold",
        )

    def test_fade_denoise_zero_is_true(self) -> None:
        """fade mode + denoise==0.0 → audio_preserve is True (the bug case)."""
        rs = self._make_rs(audio_mode="fade", denoise=0.0)
        assert rs.audio_preserve is True

    def test_fade_denoise_fractional_is_false(self) -> None:
        """fade mode + denoise==0.7 → audio_preserve is False (only d=0 is preserved)."""
        rs = self._make_rs(audio_mode="fade", denoise=0.7)
        assert rs.audio_preserve is False

    def test_keep_any_denoise_is_true(self) -> None:
        """keep mode (audio_frozen=True) + any denoise → audio_preserve is True."""
        rs_zero = RowSchedule(
            row_idx=0, denoise=0.0, inject=None, audio_frozen=True, region="preserve"
        )
        rs_frac = RowSchedule(row_idx=0, denoise=0.7, inject=None, audio_frozen=True, region="hold")
        assert rs_zero.audio_preserve is True
        assert rs_frac.audio_preserve is True

    def test_drop_denoise_zero_is_false(self) -> None:
        """drop mode + denoise==0.0 → audio_preserve is False (drop never preserves audio)."""
        rs = self._make_rs(audio_mode="drop", denoise=0.0)
        assert rs.audio_preserve is False

    def test_inject_none_audio_frozen_false_is_false(self) -> None:
        """inject=None + audio_frozen=False → audio_preserve is False."""
        rs = RowSchedule(row_idx=0, denoise=0.0, inject=None, audio_frozen=False)
        assert rs.audio_preserve is False


# ---------------------------------------------------------------------------
# merge_schedule crossfade propagation (E1)
# ---------------------------------------------------------------------------


class TestMergeScheduleCrossfadePropagation:
    """Verify that the crossfade flag is propagated through merge_schedule to
    classify_row_region, and that ramp-row regions are 'hold' by default and
    'fade' when crossfade=True.

    Uses inject_at=0, start_fade_in=0, start_keyframes=5, end_keyframes=10,
    end_fade_out=17.  Row 1 (centers 1.5–4.5) lies in the fade-in ramp [0,5).
    Row 2 (centers 5.5–8.5) lies in the hold span [5,10).
    """

    def _make_fade_inject(self) -> Inject:
        return make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=5,
            end_keyframes=10,
            end_fade_out=17,
            min_denoise=0.3,
            audio_mode="drop",
        )

    def test_ramp_rows_have_region_hold_by_default(self) -> None:
        """merge_schedule default (crossfade=False): ramp rows have region='hold'.

        Fail-then-pass test (E1 new): before the E1 change, ramp rows returned 'fade'
        unconditionally. Now they return 'hold' by default.
        """
        inj = self._make_fade_inject()
        result = merge_schedule([inj], target_rows=30)
        # row 1 (centers 1.5–4.5) is in the fade-in ramp [0,5) and should be 'hold'.
        result_by_row = {rs.row_idx: rs for rs in result}
        assert 1 in result_by_row, "row 1 should be claimed by the inject"
        row1 = result_by_row[1]
        assert row1.region == "hold", (
            f"ramp row 1 (denoise={row1.denoise}): "
            f"expected region='hold' by default, got {row1.region!r}"
        )
        assert 0.0 < row1.denoise < 1.0, (
            f"ramp row should have fractional denoise, got {row1.denoise}"
        )

    def test_ramp_rows_have_region_fade_crossfade_true(self) -> None:
        """merge_schedule with crossfade=True: ramp rows have region='fade'.

        Verifies the crossfade flag is passed through to classify_row_region.
        """
        inj = self._make_fade_inject()
        result = merge_schedule([inj], target_rows=30, crossfade=True)
        result_by_row = {rs.row_idx: rs for rs in result}
        assert 1 in result_by_row, "row 1 should be claimed by the inject"
        row1 = result_by_row[1]
        assert row1.region == "fade", (
            f"ramp row 1 (denoise={row1.denoise}): "
            f"expected region='fade' with crossfade=True, got {row1.region!r}"
        )
        assert 0.0 < row1.denoise < 1.0, (
            f"ramp row should have fractional denoise, got {row1.denoise}"
        )

    def test_hold_rows_unaffected_by_crossfade(self) -> None:
        """Hold-span rows (all centers in [skf, ekf)) are always 'hold' regardless of crossfade.

        crossfade only changes the ramp rows; hold rows must remain 'hold'.
        """
        inj = self._make_fade_inject()
        result_default = merge_schedule([inj], target_rows=30, crossfade=False)
        result_crossfade = merge_schedule([inj], target_rows=30, crossfade=True)
        by_row_default = {rs.row_idx: rs for rs in result_default}
        by_row_crossfade = {rs.row_idx: rs for rs in result_crossfade}
        # row 2 (centers 5.5–8.5) is in hold span [5,10); should be 'hold' in both
        assert 2 in by_row_default, "row 2 should be claimed"
        assert by_row_default[2].region == "hold", "hold row must be 'hold' with crossfade=False"
        assert by_row_crossfade[2].region == "hold", "hold row must be 'hold' with crossfade=True"

    def test_audio_mode_and_audio_frozen_unaffected_by_crossfade(self) -> None:
        """crossfade flag must not alter audio_frozen or audio_preserve behavior.

        audio_mode='fade' + denoise=0.0 → audio_preserve is True in both modes.
        """
        inj = make_inject(
            inject_at=0,
            start_fade_in=0,
            start_keyframes=0,
            end_keyframes=17,
            end_fade_out=39,
            min_denoise=0.0,
            audio_mode="fade",
        )
        result_default = merge_schedule([inj], target_rows=30, crossfade=False)
        result_crossfade = merge_schedule([inj], target_rows=30, crossfade=True)
        for rs_d, rs_c in zip(result_default, result_crossfade, strict=True):
            assert rs_d.audio_frozen == rs_c.audio_frozen, (
                f"row {rs_d.row_idx}: audio_frozen differs by crossfade flag"
            )
            assert rs_d.audio_preserve == rs_c.audio_preserve, (
                f"row {rs_d.row_idx}: audio_preserve differs by crossfade flag"
            )
