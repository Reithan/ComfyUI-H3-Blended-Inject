"""pytest configuration and shared fixtures for ComfyUI H3 Blended Inject tests.

Audit (2026-08-23): no helper is genuinely duplicated across two or more test
modules in identical form — the named audit candidates (make_inject, FakeNestedTensor,
row, fade_row, _fake_factory) each appear in exactly one module, or appear in two
modules with different signatures/semantics.  Nothing was extracted; this file is the
canonical home for shared fixtures if duplication arises in the future.
"""
