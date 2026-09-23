import importlib.util
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wbagent.py"


def load_module():
    spec = importlib.util.spec_from_file_location("wbagent", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# A completed turn shaped like the live TurnResponse: no `final_output`, answer
# carried on the latest assistant message (top-level `content` mirror + opaque
# `message.content` list-of-parts), preceded by user + reasoning records.
COMPLETED_TURN = {
    "id": "t-1",
    "root_turn_id": "t-1",
    "parent_turn_id": None,
    "title": "demo",
    "state": "completed",
    "created_at": "2026-06-29T20:05:56Z",
    "updated_at": "2026-06-29T20:06:00Z",
    "expires_at": "2026-06-29T20:15:56Z",
    "error_info": None,
    "agent_questions": None,
    "permission_requests": None,
    "tool_calls": [],
    "messages": [
        {"role": "user", "content": "In one short sentence, what is 2+2?",
         "message": {"role": "user", "content": "In one short sentence, what is 2+2?"}},
        {"role": "reasoning", "content": None,
         "message": {"type": "reasoning", "content": [], "summary": [], "encrypted_content": "x"}},
        {"role": "assistant", "content": "2 + 2 = 4.",
         "message": {"role": "assistant", "type": "message",
                     "content": [{"text": "2 + 2 = 4.", "type": "output_text"}]}},
    ],
}


class WbAgentCliTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def parse(self, *args):
        return self.wbagent.build_parser().parse_args(list(args))

    # Client tagging is disabled (WB_AGENT_CLIENT="") in the scope tests below so
    # they assert on the bare prompt; marker injection is covered separately.
    @mock.patch.dict(os.environ, {"WANDB_PROJECT": "project-only", "WB_AGENT_CLIENT": ""}, clear=True)
    def test_project_only_env_is_ignored_for_unscoped_create(self):
        args = self.parse("create", "hello")
        self.assertEqual(self.wbagent.create_body(args), {"user_prompt": "hello"})

    @mock.patch.dict(os.environ, {"WANDB_PROJECT": "project-from-env", "WB_AGENT_CLIENT": ""}, clear=True)
    def test_explicit_entity_pairs_with_project_env(self):
        args = self.parse("create", "--entity", "entity", "hello")
        self.assertEqual(
            self.wbagent.create_body(args),
            {"user_prompt": "hello", "entity": "entity", "project": "project-from-env"},
        )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_explicit_project_still_requires_entity(self):
        args = self.parse("create", "--project", "project", "hello")
        with self.assertRaisesRegex(SystemExit, "entity"):
            self.wbagent.create_body(args)

    def test_bad_prompt_parts_json_exits_cleanly(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write("{bad json")
            handle.flush()
            args = self.parse("create", "--prompt-parts", handle.name)
            with self.assertRaisesRegex(SystemExit, "valid JSON"):
                self.wbagent.load_prompt(args)


class CreateBodyTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def parse(self, *args):
        return self.wbagent.build_parser().parse_args(list(args))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_agent_config_override_in_body(self):
        args = self.parse("create", "--agent-config-override", "candidate", "hi")
        self.assertEqual(self.wbagent.create_body(args)["agent_config_override"], "candidate")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_allowed_network_domains_repeatable_and_additive(self):
        args = self.parse(
            "create",
            "--allow-all-network-egress",
            "--allowed-network-domain", "pypi.org",
            "--allowed-network-domain", "files.pythonhosted.org",
            "hi",
        )
        perms = self.wbagent.create_body(args)["permissions"]
        self.assertEqual(perms["allow_all_network_egress"], True)
        self.assertEqual(perms["allowed_network_domains"], ["pypi.org", "files.pythonhosted.org"])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_title_and_parent_and_no_permissions_by_default(self):
        args = self.parse("create", "--title", "T", "--parent-turn-id", "p-1", "hi")
        body = self.wbagent.create_body(args)
        self.assertEqual(body["title"], "T")
        self.assertEqual(body["parent_turn_id"], "p-1")
        self.assertNotIn("permissions", body)
        self.assertNotIn("agent_config_override", body)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def test_assistant_text_from_top_level_mirror(self):
        self.assertEqual(self.wbagent.assistant_text(COMPLETED_TURN), "2 + 2 = 4.")

    def test_assistant_text_walks_list_parts_when_mirror_dropped(self):
        # Regression: with no top-level mirror, extraction must NOT echo the user
        # prompt or reasoning record — it must walk the assistant message parts.
        turn = {
            "messages": [
                {"role": "user", "content": None, "message": {"content": "ask"}},
                {"role": "reasoning", "content": None, "message": {"content": []}},
                {"role": "assistant", "content": None,
                 "message": {"content": [{"type": "output_text", "text": "the real answer"}]}},
            ]
        }
        self.assertEqual(self.wbagent.assistant_text(turn), "the real answer")

    def test_assistant_text_never_returns_user_prompt(self):
        turn = {"messages": [{"role": "user", "content": "secret prompt", "message": {"content": "secret prompt"}}]}
        self.assertEqual(self.wbagent.assistant_text(turn), "")

    def test_turn_summary_surfaces_live_fields(self):
        summary = self.wbagent.turn_summary(COMPLETED_TURN)
        self.assertEqual(summary["title"], "demo")
        self.assertTrue(summary["has_response"])
        self.assertNotIn("has_final_output", summary)
        self.assertEqual(summary["agent_questions"], 0)
        self.assertEqual(summary["permission_requests"], 0)
        self.assertIn("expires_at", summary)

    def test_interaction_lines_for_agent_question(self):
        turn = {
            "messages": [],
            "agent_questions": [{"type": "multiple_choice", "question": "Which run?", "options": ["a", "b"]}],
        }
        lines = self.wbagent.interaction_lines(turn)
        self.assertTrue(any("AGENT QUESTION [0]: Which run?" in line for line in lines))
        self.assertTrue(any("0. a" in line for line in lines))

    def test_interaction_lines_for_network_request(self):
        turn = {
            "messages": [],
            "permission_requests": [{"type": "network_access", "reason": "fetch pkg", "domains": ["pypi.org"]}],
        }
        lines = self.wbagent.interaction_lines(turn)
        self.assertTrue(any("NETWORK ACCESS REQUESTED: fetch pkg -> [pypi.org]" in line for line in lines))


class WaitLogicTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def _wait_args(self):
        ns = mock.Mock()
        ns.since_updated_at = None
        ns.since_state = None
        ns.since_message_count = None
        return ns

    def test_changed_true_when_assistant_text_present(self):
        self.assertTrue(self.wbagent.changed(COMPLETED_TURN, self._wait_args()))

    def test_changed_true_on_agent_question(self):
        turn = {"state": "in_progress", "messages": [], "updated_at": "x",
                "agent_questions": [{"type": "multiple_choice", "question": "?", "options": []}]}
        self.assertTrue(self.wbagent.changed(turn, self._wait_args()))

    def test_changed_false_when_idle(self):
        turn = {"state": "in_progress", "messages": [], "updated_at": "x"}
        self.assertFalse(self.wbagent.changed(turn, self._wait_args()))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def parse(self, *args):
        return self.wbagent.build_parser().parse_args(list(args))

    @mock.patch.dict(os.environ, {"WANDB_PROJECT": "p"}, clear=True)
    def test_query_scope_project_only_env_dropped(self):
        args = self.parse("query")
        self.assertEqual(self.wbagent.resolve_query_scope(args), (None, None))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_query_scope_entity_without_project_raises(self):
        args = self.parse("query", "--entity", "e")
        with self.assertRaisesRegex(SystemExit, "both entity and project"):
            self.wbagent.resolve_query_scope(args)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_query_scope_pair(self):
        args = self.parse("query", "--entity", "e", "--project", "p")
        self.assertEqual(self.wbagent.resolve_query_scope(args), ("e", "p"))


class AuthAndHttpTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    @mock.patch.dict(os.environ, {"WANDB_API_KEY": "k"}, clear=True)
    def test_auth_header_defaults_username_to_api(self):
        import base64
        header = self.wbagent.auth_header()
        self.assertEqual(header, "Basic " + base64.b64encode(b"api:k").decode())

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_auth_header_missing_raises_when_required(self):
        with self.assertRaises(self.wbagent.ApiError):
            self.wbagent.auth_header(required=True)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_auth_header_optional_returns_none(self):
        self.assertIsNone(self.wbagent.auth_header(required=False))

    def test_request_json_maps_http_error_to_apierror(self):
        err = urllib.error.HTTPError(
            "https://x/api/v1/health", 422, "Unprocessable", {}, io.BytesIO(b'{"detail":"bad"}')
        )
        with mock.patch.object(self.wbagent.urllib.request, "urlopen", side_effect=err):
            with self.assertRaises(self.wbagent.ApiError) as ctx:
                self.wbagent.request_json("GET", "/api/v1/health", auth=False)
        self.assertEqual(ctx.exception.status, 422)


class ClientAttributionTests(unittest.TestCase):
    def setUp(self):
        self.wbagent = load_module()

    def parse(self, *args):
        return self.wbagent.build_parser().parse_args(list(args))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_string_prompt_gets_default_client_marker(self):
        args = self.parse("create", "hello", "world")
        prompt = self.wbagent.create_body(args)["user_prompt"]
        self.assertEqual(prompt[0], {"type": "client_info", "client": "coding_agent"})
        self.assertEqual(prompt[1], {"type": "text", "text": "hello world"})

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_client_flag_overrides_tag(self):
        args = self.parse("create", "--client", "codex_cli", "hi")
        self.assertEqual(self.wbagent.create_body(args)["user_prompt"][0]["client"], "codex_cli")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_empty_client_flag_disables_tagging(self):
        args = self.parse("create", "--client", "", "hi")
        self.assertEqual(self.wbagent.create_body(args)["user_prompt"], "hi")

    @mock.patch.dict(os.environ, {"WB_AGENT_CLIENT": "from_env"}, clear=True)
    def test_env_sets_tag(self):
        args = self.parse("create", "hi")
        self.assertEqual(self.wbagent.create_body(args)["user_prompt"][0]["client"], "from_env")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_marker_prepended_to_prompt_parts_list(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            handle.write('[{"type": "text", "text": "explain"}]')
            handle.flush()
            path = handle.name
        try:
            args = self.parse("create", "--prompt-parts", path)
            prompt = self.wbagent.create_body(args)["user_prompt"]
            self.assertEqual(prompt[0], {"type": "client_info", "client": "coding_agent"})
            self.assertEqual(prompt[1], {"type": "text", "text": "explain"})
        finally:
            os.unlink(path)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_request_json_sends_client_headers(self):
        captured = {}

        class FakeResp:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, *a, **k):
            captured["headers"] = dict(req.headers)
            return FakeResp()

        with mock.patch.object(self.wbagent.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.wbagent.request_json("GET", "/api/v1/health", auth=False)
        # urllib title-cases header keys
        self.assertEqual(captured["headers"].get("X-wandb-client"), "coding_agent")
        self.assertIn("coding_agent", captured["headers"].get("User-agent", ""))

    @mock.patch.dict(os.environ, {"WB_AGENT_CLIENT": ""}, clear=True)
    def test_disabled_client_omits_headers(self):
        captured = {}

        class FakeResp:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, *a, **k):
            captured["headers"] = dict(req.headers)
            return FakeResp()

        with mock.patch.object(self.wbagent.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.wbagent.request_json("GET", "/api/v1/health", auth=False)
        self.assertNotIn("X-wandb-client", captured["headers"])


if __name__ == "__main__":
    unittest.main()
