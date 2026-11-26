from rag_agent.overlay_state import OverlaySnapshot, current, enqueue, take, update


def test_native_commands_only_reach_the_current_browser_page():
    update(OverlaySnapshot(client_id="page-a", running=True, ai_enabled=False))
    enqueue("ai")
    assert take("page-b") == []
    assert take("page-a") == ["ai"]
    assert take("page-a") == []

    enqueue("assess")
    update(OverlaySnapshot(client_id="page-b", running=True))
    assert take("page-b") == []
    assert current()["snapshot"]["client_id"] == "page-b"
