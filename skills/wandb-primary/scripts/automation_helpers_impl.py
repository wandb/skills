# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Automation list/create helpers.

List and create W&B Automations from the command line.

    H=skills/wandb-primary/scripts/automation_helpers.py
    python $H list --entity E --project P
    python $H list --registry NAME --org ORG
    python $H create --name NAME --event ADD_ARTIFACT_ALIAS --action NOTIFICATION --slack-integration-id ID --message '...'
    python $H --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from wandb_run_ops_lib.common import api, execute_graphql, resolve_path, split_path
from wandb_run_ops_lib.core import emit

_REGISTRY_PREFIX = "wandb-registry-"
_GRAPHQL_PAGE_SIZE = 50
_EMPTY_ARTIFACT_FILTER = {"$or": [{"$and": []}]}
_EMPTY_RUN_FILTER = {"$and": []}

_TRIGGER_FIELDS = """
id name description enabled createdAt lastExecutedAt
createdBy { username }
scope {
  __typename
  ... on Project { id name entityName }
  ... on ArtifactSequence { id name }
  ... on ArtifactPortfolio { id name }
  ... on Entity { id name }
}
triggeringCondition {
  ... on FilterEventTriggeringCondition { eventType filter }
}
triggeredAction {
  __typename
  ... on PushNotificationTriggeredAction { title body user { id username } }
  ... on NotificationTriggeredAction { title message integration { id } }
  ... on GenericWebhookTriggeredAction { requestPayload integration { id } }
}
"""

_PROJECT_TRIGGERS_QUERY = f"""
query PublicProjectAutomations(
  $entity: String!, $project: String!, $projectOnly: Boolean
) {{
  project(name: $project, entityName: $entity) {{
    id name entityName
    triggers(projectOnly: $projectOnly) {{
      {_TRIGGER_FIELDS}
    }}
  }}
}}
"""

_REGISTRY_PROJECT_QUERY = """
query PublicRegistryProject($orgName: String!, $projectName: String!) {
  organization(name: $orgName) {
    orgEntity {
      name
      project(name: $projectName) { id name entityName }
    }
  }
}
"""

_PROJECT_ID_QUERY = """
query PublicProjectId($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) { id name entityName }
}
"""

_COLLECTION_TRIGGERS_QUERY = f"""
query PublicCollectionAutomations($id: ID!, $after: String, $first: Int!) {{
  artifactCollection(id: $id) {{
    id name
    triggers(first: $first, after: $after) {{
      totalCount
      pageInfo {{ hasNextPage endCursor }}
      edges {{ node {{ {_TRIGGER_FIELDS} }} }}
    }}
  }}
}}
"""

_TRIGGER_QUERY = f"""
query PublicTrigger($id: ID!) {{
  trigger(id: $id) {{ {_TRIGGER_FIELDS} }}
}}
"""

_INTEGRATIONS_QUERY = """
query PublicTeamIntegrations($entity: String!, $first: Int!, $after: String) {
  entity(name: $entity) {
    integrations(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges {
        node {
          __typename
          ... on SlackIntegration { id teamName channelName }
          ... on GenericWebhookIntegration { id name urlEndpoint }
        }
      }
    }
  }
}
"""

_TEMPLATE_VARS_QUERY = """
query PublicTriggerTemplateVars($eventType: EventTriggeringConditionType) {
  triggerTemplateVariables(eventType: $eventType) { name description }
}
"""

_VIEWER_ID_QUERY = """
query PublicViewerId { viewer { id username } }
"""

_CREATE_MUTATION = """
mutation PublicCreateAutomation($input: CreateFilterTriggerInput!) {
  createFilterTrigger(input: $input) { trigger { id name enabled } }
}
"""

_ACTION_TYPES = (
    "NOTIFICATION",
    "GENERIC_WEBHOOK",
    "PUSH_NOTIFICATION",
)
_RUN_STATES = ("RUNNING", "FINISHED", "FAILED")
_CMP_OPS = ("$gt", "$gte", "$lt", "$lte", "$eq", "$ne")
_AGG_OPS = ("AVERAGE", "MIN", "MAX")


def registry_project_name(name: str) -> str:
    """Return the backing project name for a Registry."""
    name = name.strip()
    if name.startswith(_REGISTRY_PREFIX):
        return name
    return f"{_REGISTRY_PREFIX}{name}"


def is_registry_project(project: str | None) -> bool:
    """Return whether a project name is a Registry backing project."""
    return bool(project) and project.startswith(_REGISTRY_PREFIX) and project != _REGISTRY_PREFIX


def requested_project(args: argparse.Namespace) -> str | None:
    """CLI --project, else WANDB_PROJECT."""
    return args.project or os.environ.get("WANDB_PROJECT")


def is_registry_scope(args: argparse.Namespace) -> bool:
    """Return whether list/create should use registry-wide PROJECT scope."""
    return bool(args.registry) or is_registry_project(requested_project(args))


def artifact_event_filter(*, alias_regex: str | None = None, tag_regex: str | None = None) -> str:
    """JSON eventFilter for artifact alias/tag events."""
    clauses: list[dict[str, Any]] = []
    if alias_regex:
        clauses.append({"alias": {"$regex": alias_regex}})
    if tag_regex:
        clauses.append({"tag": {"$regex": tag_regex}})
    tree = {"$or": [{"$and": clauses}]} if clauses else _EMPTY_ARTIFACT_FILTER
    return json.dumps(tree, separators=(",", ":"))


def run_filter_json(
    *,
    sweep: str | None = None,
    run_name_regex: str | None = None,
    username: str | None = None,
) -> str:
    """Mongo run_filter string nested inside run-event eventFilter."""
    clauses: list[dict[str, Any]] = []
    if sweep:
        clauses.append({"sweep": sweep})
    if run_name_regex:
        clauses.append({"displayName": {"$regex": run_name_regex}})
    if username:
        clauses.append({"username": username})
    return json.dumps({"$and": clauses} if clauses else _EMPTY_RUN_FILTER, separators=(",", ":"))


def run_state_event_filter(
    states: list[str],
    *,
    sweep: str | None = None,
    run_name_regex: str | None = None,
    username: str | None = None,
) -> str:
    """JSON eventFilter for RUN_STATE."""
    return json.dumps(
        {
            "run_filter": run_filter_json(
                sweep=sweep, run_name_regex=run_name_regex, username=username
            ),
            "run_state_filter": {"states": states},
        },
        separators=(",", ":"),
    )


def run_metric_threshold_filter(
    *,
    name: str,
    threshold: float,
    window_size: int = 5,
    cmp_op: str = "$gt",
    agg_op: str = "AVERAGE",
    sweep: str | None = None,
    run_name_regex: str | None = None,
    username: str | None = None,
) -> str:
    """JSON eventFilter for RUN_METRIC threshold."""
    return json.dumps(
        {
            "run_filter": run_filter_json(
                sweep=sweep, run_name_regex=run_name_regex, username=username
            ),
            "run_metric_filter": {
                "threshold_filter": {
                    "name": name,
                    "threshold": threshold,
                    "window_size": window_size,
                    "cmp_op": cmp_op,
                    "agg_op": agg_op,
                }
            },
        },
        separators=(",", ":"),
    )


def compact_trigger(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Flatten a Trigger node into a compact row."""
    if not node:
        return None
    scope = node.get("scope") or {}
    condition = node.get("triggeringCondition") or {}
    action = node.get("triggeredAction") or {}
    created_by = node.get("createdBy") or {}
    action_type = (action.get("__typename") or "").removesuffix("TriggeredAction")
    return {
        "id": node.get("id"),
        "name": node.get("name"),
        "description": node.get("description"),
        "enabled": node.get("enabled"),
        "created_at": node.get("createdAt"),
        "last_executed_at": node.get("lastExecutedAt"),
        "created_by": created_by.get("username"),
        "event": condition.get("eventType"),
        "filter": condition.get("filter"),
        "action_type": action_type or None,
        "action": _compact_action(action),
        "scope_type": scope.get("__typename"),
        "scope_id": scope.get("id"),
        "scope_name": scope.get("name"),
        "scope_entity": scope.get("entityName"),
    }


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("__typename")
    if kind == "NotificationTriggeredAction":
        integration = action.get("integration") or {}
        return {
            "title": action.get("title"),
            "message": action.get("message"),
            "integration_id": integration.get("id"),
        }
    if kind == "GenericWebhookTriggeredAction":
        integration = action.get("integration") or {}
        return {
            "request_payload": action.get("requestPayload"),
            "integration_id": integration.get("id"),
        }
    if kind == "PushNotificationTriggeredAction":
        user = action.get("user") or {}
        return {
            "title": action.get("title"),
            "body": action.get("body"),
            "user": user.get("username"),
            "user_id": user.get("id"),
        }
    return {}


def resolve_registry_project(
    wb: Any, *, registry: str, org: str | None = None, entity: str | None = None
) -> dict[str, str]:
    """Resolve a registry name to org-entity + wandb-registry-* project ids."""
    project_name = registry_project_name(registry)
    if org:
        data = execute_graphql(
            wb,
            _REGISTRY_PROJECT_QUERY,
            {"orgName": org, "projectName": project_name},
        )
        org_entity = (data.get("organization") or {}).get("orgEntity") or {}
        project = org_entity.get("project") or {}
        entity_name = org_entity.get("name")
        if not project.get("id") or not entity_name:
            raise SystemExit(
                f"Registry project {project_name!r} not found under organization {org!r}"
            )
        return {
            "id": project["id"],
            "name": project.get("name") or project_name,
            "entityName": project.get("entityName") or entity_name,
        }

    entity_name = entity or os.environ.get("WANDB_ENTITY")
    if not entity_name:
        raise SystemExit(
            "Pass --org ORG (from /orgs/{org}/registry/...) or --entity for the org entity"
        )
    data = execute_graphql(
        wb, _PROJECT_ID_QUERY, {"entity": entity_name, "project": project_name}
    )
    project = data.get("project") or {}
    if not project.get("id"):
        raise SystemExit(
            f"Registry project {project_name!r} not found on entity {entity_name!r}; "
            "pass --org with the organization name from the registry URL"
        )
    return {
        "id": project["id"],
        "name": project.get("name") or project_name,
        "entityName": project.get("entityName") or entity_name,
    }


def resolve_project(wb: Any, entity: str | None, project: str | None) -> dict[str, str]:
    """Resolve a Models project to its GraphQL id."""
    path = resolve_path(entity, project)
    entity_name, project_name = split_path(path)
    data = execute_graphql(
        wb, _PROJECT_ID_QUERY, {"entity": entity_name, "project": project_name}
    )
    node = data.get("project") or {}
    if not node.get("id"):
        raise SystemExit(f"Project {path} not found")
    return {
        "id": node["id"],
        "name": node.get("name") or project_name,
        "entityName": node.get("entityName") or entity_name,
    }


def list_project_automations(
    wb: Any, *, entity: str, project: str, project_only: bool
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """List automations on a project (or registry backing project)."""
    data = execute_graphql(
        wb,
        _PROJECT_TRIGGERS_QUERY,
        {"entity": entity, "project": project, "projectOnly": project_only},
    )
    project_node = data.get("project") or {}
    if not project_node.get("id"):
        raise SystemExit(f"Project {entity}/{project} not found")
    rows = [
        row
        for node in (project_node.get("triggers") or [])
        if (row := compact_trigger(node))
    ]
    meta = {
        "id": project_node["id"],
        "name": project_node.get("name") or project,
        "entityName": project_node.get("entityName") or entity,
    }
    return meta, rows


def list_collection_automations(wb: Any, collection_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """List automations whose scope is exactly this collection (paginated)."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    name = ""
    total = 0
    while True:
        variables: dict[str, Any] = {"id": collection_id, "first": _GRAPHQL_PAGE_SIZE}
        if cursor:
            variables["after"] = cursor
        data = execute_graphql(wb, _COLLECTION_TRIGGERS_QUERY, variables)
        collection = data.get("artifactCollection") or {}
        if not collection.get("id"):
            raise SystemExit(f"Collection {collection_id} not found")
        name = collection.get("name") or name
        conn = collection.get("triggers") or {}
        total = int(conn.get("totalCount") or 0)
        for edge in conn.get("edges") or []:
            row = compact_trigger(edge.get("node"))
            if row:
                rows.append(row)
        page = conn.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        if not cursor:
            break
    meta = {"id": collection_id, "name": name, "totalCount": str(total)}
    return meta, rows


def get_automation(wb: Any, trigger_id: str) -> dict[str, Any]:
    """Fetch one automation by GraphQL id (use after create; project lists can lag)."""
    data = execute_graphql(wb, _TRIGGER_QUERY, {"id": trigger_id})
    row = compact_trigger(data.get("trigger"))
    if not row:
        raise SystemExit(f"Automation {trigger_id} not found")
    return row


def _compact_integration(node: dict[str, Any]) -> dict[str, Any] | None:
    kind = node.get("__typename")
    if kind == "SlackIntegration":
        return {
            "type": "slack",
            "id": node.get("id"),
            "team": node.get("teamName"),
            "channel": node.get("channelName"),
        }
    if kind == "GenericWebhookIntegration":
        return {
            "type": "webhook",
            "id": node.get("id"),
            "name": node.get("name"),
            "url": node.get("urlEndpoint"),
        }
    return None


def list_integrations(
    wb: Any, entity: str, *, first: int = _GRAPHQL_PAGE_SIZE
) -> list[dict[str, Any]]:
    """List Slack and webhook destinations on a team entity (paginated)."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        variables: dict[str, Any] = {"entity": entity, "first": first}
        if cursor:
            variables["after"] = cursor
        data = execute_graphql(wb, _INTEGRATIONS_QUERY, variables)
        entity_node = data.get("entity") or {}
        conn = entity_node.get("integrations") or {}
        for edge in conn.get("edges") or []:
            row = _compact_integration(edge.get("node") or {})
            if row:
                rows.append(row)
        page = conn.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        if not cursor:
            break
    return rows


def template_variables(wb: Any, event_type: str | None = None) -> list[dict[str, str]]:
    """Return triggerTemplateVariables for an event type."""
    data = execute_graphql(wb, _TEMPLATE_VARS_QUERY, {"eventType": event_type})
    return [
        {"name": item.get("name"), "description": item.get("description")}
        for item in (data.get("triggerTemplateVariables") or [])
        if item.get("name")
    ]


def build_action_config(
    action: str,
    *,
    slack_integration_id: str | None = None,
    title: str | None = None,
    message: str | None = None,
    webhook_integration_id: str | None = None,
    payload: str | None = None,
    user_id: str | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    """Build triggeredActionConfig for createFilterTrigger."""
    if action == "NOTIFICATION":
        if not slack_integration_id:
            raise SystemExit("--slack-integration-id is required for --action NOTIFICATION")
        return {
            "notificationActionInput": {
                "integrationID": slack_integration_id,
                "title": title,
                "message": message,
            }
        }
    if action == "GENERIC_WEBHOOK":
        if not webhook_integration_id:
            raise SystemExit(
                "--webhook-integration-id is required for --action GENERIC_WEBHOOK"
            )
        return {
            "genericWebhookActionInput": {
                "integrationID": webhook_integration_id,
                "requestPayload": payload,
            }
        }
    if action == "PUSH_NOTIFICATION":
        if not user_id:
            raise SystemExit("--user-id is required for --action PUSH_NOTIFICATION")
        return {
            "pushNotificationActionInput": {
                "userID": user_id,
                "title": title or "",
                "body": body or message or "",
            }
        }
    raise SystemExit(f"Unsupported action {action!r}")


def event_filter_from_args(args: argparse.Namespace) -> str:
    """Build eventFilter JSON from CLI flags."""
    if args.event_filter:
        return args.event_filter
    event = args.event
    if event in {
        "ADD_ARTIFACT_ALIAS",
        "ADD_ARTIFACT_TAG",
        "REMOVE_ARTIFACT_TAG",
        "CREATE_ARTIFACT",
        "LINK_MODEL",
        "UNLINK_ARTIFACT",
        "ADD_COLLECTION_TAG",
        "REMOVE_COLLECTION_TAG",
    }:
        return artifact_event_filter(
            alias_regex=args.alias_regex, tag_regex=args.tag_regex
        )
    if event == "RUN_STATE":
        states = list(args.run_state or [])
        if not states:
            raise SystemExit("--run-state is required for --event RUN_STATE")
        return run_state_event_filter(
            states,
            sweep=args.sweep,
            run_name_regex=args.run_name_regex,
            username=args.username,
        )
    if event == "RUN_METRIC":
        if not args.metric_name or args.threshold is None:
            raise SystemExit("--metric-name and --threshold are required for RUN_METRIC")
        return run_metric_threshold_filter(
            name=args.metric_name,
            threshold=args.threshold,
            window_size=args.window_size,
            cmp_op=args.cmp_op,
            agg_op=args.agg_op,
            sweep=args.sweep,
            run_name_regex=args.run_name_regex,
            username=args.username,
        )
    raise SystemExit(
        f"Pass --event-filter JSON for {event}; helper builders cover alias/tag, "
        "RUN_STATE, and RUN_METRIC"
    )


def create_automation(
    wb: Any,
    *,
    name: str,
    event: str,
    event_filter: str,
    action: str,
    action_config: dict[str, Any],
    scope_type: str,
    scope_id: str,
    description: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Create an automation and re-read it by id."""
    payload = execute_graphql(
        wb,
        _CREATE_MUTATION,
        {
            "input": {
                "name": name,
                "description": description,
                "triggeringEventType": event,
                "scopeType": scope_type,
                "scopeID": scope_id,
                "eventFilter": event_filter,
                "triggeredActionType": action,
                "triggeredActionConfig": action_config,
                "enabled": enabled,
            }
        },
    )
    created = ((payload.get("createFilterTrigger") or {}).get("trigger")) or {}
    trigger_id = created.get("id")
    if not trigger_id:
        raise SystemExit("createFilterTrigger returned no trigger id")
    try:
        row = get_automation(wb, trigger_id)
    except SystemExit:
        return {
            "id": trigger_id,
            "name": created.get("name") or name,
            "enabled": created.get("enabled"),
            "verified": False,
        }
    row["verified"] = True
    return row


def cmd_list(args: argparse.Namespace) -> None:
    wb = api()
    if args.collection_id:
        meta, rows = list_collection_automations(wb, args.collection_id)
        kind = "collection"
        project_only = False
        print(f"scope=collection id={meta['id']} name={meta.get('name')}")
        print(
            "limit_caveat=collection list is exact-scope only; registry-wide "
            "automations are on python $H list --registry NAME --org ORG"
        )
    elif is_registry_scope(args):
        registry = args.registry or (requested_project(args) or "").removeprefix(
            _REGISTRY_PREFIX
        )
        project = resolve_registry_project(
            wb, registry=registry, org=args.org, entity=args.entity
        )
        project_only = not args.include_collections
        _meta, rows = list_project_automations(
            wb,
            entity=project["entityName"],
            project=project["name"],
            project_only=project_only,
        )
        kind = "registry"
        print(
            f"scope=registry name={registry} project={project['entityName']}/{project['name']} "
            f"id={project['id']} project_only={str(project_only).lower()}"
        )
    else:
        project = resolve_project(wb, args.entity, args.project)
        project_only = bool(args.project_only)
        _meta, rows = list_project_automations(
            wb,
            entity=project["entityName"],
            project=project["name"],
            project_only=project_only,
        )
        kind = "project"
        print(
            f"scope=project path={project['entityName']}/{project['name']} "
            f"id={project['id']} project_only={str(project_only).lower()}"
        )

    print(f"kind={kind}")
    print(f"count={len(rows)}")
    emit("sample_scope", {"kind": kind, "count": len(rows), "project_only": project_only if kind != "collection" else False})
    for row in rows:
        emit("automation_row", row)
    print(f"answer=count={len(rows)} kind={kind}")
    print(
        "interpretation_hint=Ground claims on emitted automation_row lines. "
        "Registry-wide automations are PROJECT-scoped on wandb-registry-*; "
        "collection-scoped automations are a separate list."
    )


def cmd_get(args: argparse.Namespace) -> None:
    row = get_automation(api(), args.id)
    emit("automation_row", row)
    print(f"answer=id={row['id']} name={row.get('name')} enabled={row.get('enabled')}")


def cmd_integrations(args: argparse.Namespace) -> None:
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        raise SystemExit("Pass --entity for the team that owns Slack/webhook integrations")
    rows = list_integrations(api(), entity)
    print(f"entity={entity}")
    print(f"count={len(rows)}")
    for row in rows:
        emit("integration_row", row)
    print(f"answer=count={len(rows)}")


def cmd_template_vars(args: argparse.Namespace) -> None:
    rows = template_variables(api(), args.event)
    print(f"event={args.event or 'all'}")
    print(f"count={len(rows)}")
    for row in rows:
        emit("template_variable", row)
    print(f"answer=count={len(rows)}")


def cmd_create(args: argparse.Namespace) -> None:
    wb = api()
    if args.collection_id:
        scope_type, scope_id = "ARTIFACT_COLLECTION", args.collection_id
    elif is_registry_scope(args):
        registry = args.registry or (requested_project(args) or "").removeprefix(
            _REGISTRY_PREFIX
        )
        project = resolve_registry_project(
            wb, registry=registry, org=args.org, entity=args.entity
        )
        scope_type, scope_id = "PROJECT", project["id"]
    else:
        project = resolve_project(wb, args.entity, args.project)
        scope_type, scope_id = "PROJECT", project["id"]

    user_id = args.user_id
    if args.action == "PUSH_NOTIFICATION" and not user_id:
        viewer = execute_graphql(wb, _VIEWER_ID_QUERY, {})
        user_id = (viewer.get("viewer") or {}).get("id")
    action_config = build_action_config(
        args.action,
        slack_integration_id=args.slack_integration_id,
        title=args.title,
        message=args.message,
        webhook_integration_id=args.webhook_integration_id,
        payload=args.payload,
        user_id=user_id,
        body=args.body,
    )
    event_filter = event_filter_from_args(args)
    row = create_automation(
        wb,
        name=args.name,
        event=args.event,
        event_filter=event_filter,
        action=args.action,
        action_config=action_config,
        scope_type=scope_type,
        scope_id=scope_id,
        description=args.description,
        enabled=not args.disabled,
    )
    emit("automation_row", row)
    verified = row.get("verified", True)
    print(
        f"answer=created id={row.get('id')} name={row.get('name')} "
        f"enabled={row.get('enabled')} verified={verified}"
    )


def _add_scope_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity", help="Team or org entity (or WANDB_ENTITY).")
    parser.add_argument("--project", help="Models project, or wandb-registry-* (or WANDB_PROJECT).")
    parser.add_argument("--org", help="Organization name from /orgs/{org}/registry/{registry}.")
    parser.add_argument("--registry", help="Registry display name (nice-overview) or wandb-registry-* project.")
    parser.add_argument("--collection-id", help="GraphQL Artifact collection/portfolio id.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and create W&B Automations (GraphQL createFilterTrigger).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="List automations for a project, registry, or collection.")
    _add_scope_flags(listing)
    listing.add_argument(
        "--project-only",
        action="store_true",
        help="On a Models project, skip collection-scoped automations.",
    )
    listing.add_argument(
        "--include-collections",
        action="store_true",
        help="On a registry, also include collection-scoped automations (default is registry-wide only).",
    )
    listing.set_defaults(func=cmd_list)

    get_cmd = sub.add_parser("get", help="Fetch one automation by GraphQL id.")
    get_cmd.add_argument("--id", required=True, help="Trigger GraphQL id.")
    get_cmd.set_defaults(func=cmd_get)

    integrations = sub.add_parser("integrations", help="List Slack and webhook destinations for a team.")
    integrations.add_argument("--entity", help="Team entity (or WANDB_ENTITY).")
    integrations.set_defaults(func=cmd_integrations)

    tvars = sub.add_parser("template-vars", help="List triggerTemplateVariables for an event.")
    tvars.add_argument("--event", help="EventTriggeringConditionType, e.g. ADD_ARTIFACT_ALIAS.")
    tvars.set_defaults(func=cmd_template_vars)

    create = sub.add_parser("create", help="Create an automation via createFilterTrigger, then re-read by id.")
    _add_scope_flags(create)
    create.add_argument("--name", required=True)
    create.add_argument("--description")
    create.add_argument("--event", required=True, help="e.g. ADD_ARTIFACT_ALIAS, RUN_STATE, RUN_METRIC.")
    create.add_argument("--action", required=True, choices=list(_ACTION_TYPES))
    create.add_argument("--slack-integration-id")
    create.add_argument("--webhook-integration-id")
    create.add_argument("--payload", help="Webhook JSON payload string.")
    create.add_argument("--title")
    create.add_argument("--message")
    create.add_argument("--body")
    create.add_argument("--user-id", help="Push notification user GraphQL id (defaults to viewer).")
    create.add_argument("--alias-regex")
    create.add_argument("--tag-regex")
    create.add_argument("--run-state", action="append", choices=list(_RUN_STATES), default=[])
    create.add_argument("--sweep", help="Restrict RUN_* events to child runs of this sweep id.")
    create.add_argument("--run-name-regex")
    create.add_argument("--username")
    create.add_argument("--metric-name")
    create.add_argument("--threshold", type=float)
    create.add_argument("--window-size", type=int, default=5)
    create.add_argument("--cmp-op", choices=list(_CMP_OPS), default="$gt")
    create.add_argument("--agg-op", choices=list(_AGG_OPS), default="AVERAGE")
    create.add_argument("--event-filter", help="Raw eventFilter JSON string (overrides builders).")
    create.add_argument("--disabled", action="store_true")
    create.set_defaults(func=cmd_create)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        emit("error", {"type": type(exc).__name__, "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
