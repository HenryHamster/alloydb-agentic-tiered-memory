import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from retain.entities import FuzzyEntityResolver, mention_key
from retain.linking import GeminiSupersedeClassifier, StoreLinker
from retain.models import (
    Category,
    ClaimType,
    Entity,
    EntityMention,
    EntityType,
    FactType,
    MemoryFact,
    Relation,
    Scope,
    Speaker,
    UpdateAction,
    UpdateDecision,
    Visibility,
    stable_id,
)

AT = datetime(2026, 10, 20, tzinfo=timezone.utc)
SCOPE = Scope("user-stage", "project-a", "session-a")


def entity(name, entity_type=EntityType.ORGANIZATION, *, canonical=None, aliases=(), user=None):
    canonical = canonical or name.casefold()
    return Entity(
        stable_id("entity", user or SCOPE.user_id, entity_type.value, canonical),
        user or SCOPE.user_id,
        name,
        canonical,
        entity_type,
        tuple(aliases),
    )


class EntityStore:
    def __init__(self, entities):
        self.entities = entities
        self.queries = []

    def find_entities(self, user_id, names):
        self.queries.append((user_id, tuple(names)))
        return self.entities


def fact(
    name,
    *,
    fact_id=None,
    scope=SCOPE,
    visibility=Visibility.PROJECT,
    category=Category.EVENT,
    claim_type=ClaimType.ASSERTED,
    speaker=Speaker.USER,
    embedding=(1.0, 0.0),
    occurred_at=None,
    entity_ids=(),
    valid_to=None,
):
    return MemoryFact(
        fact_id=fact_id or stable_id("fact", name, scope.user_id, scope.project_id),
        scope=scope,
        visibility=visibility,
        fact_type=FactType.WORLD,
        category=category,
        speaker=speaker,
        claim_type=claim_type,
        text=name,
        evidence=name,
        source_document_id=stable_id("document", name),
        source_episode_id="episode",
        recorded_at=AT,
        valid_from=AT,
        occurred_at=occurred_at,
        valid_to=valid_to,
        embedding=embedding,
        entity_ids=tuple(entity_ids),
    )


class LinkStore:
    def __init__(self, similar=(), near=()):
        self.similar = list(similar)
        self.near = list(near)
        self.similarity_calls = []
        self.temporal_calls = []

    def similar_facts(self, reader, embedding, **kwargs):
        self.similarity_calls.append((reader, kwargs))
        return self.similar

    def facts_near_time(self, reader, at, window, **kwargs):
        self.temporal_calls.append((reader, at, window, kwargs))
        return self.near


def test_entity_resolution_reuses_exact_alias_and_only_adds_supported_alias():
    known = entity("Database Platform", canonical="database platform", aliases=("db system",))
    mention = EntityMention("DB systems", EntityType.ORGANIZATION)
    resolver = FuzzyEntityResolver(EntityStore([known]))

    resolution = resolver.resolve(SCOPE.user_id, [mention])

    assert resolution.entity_ids[mention_key(mention)] == known.entity_id
    assert len(resolution.upserts) == 1
    assert resolution.upserts[0].aliases == ("db system", "DB systems")

    exact = EntityMention("DB SYSTEM", EntityType.ORGANIZATION)
    exact_resolution = FuzzyEntityResolver(EntityStore([known])).resolve(SCOPE.user_id, [exact])
    assert exact_resolution.entity_ids[mention_key(exact)] == known.entity_id
    assert exact_resolution.upserts == ()


def test_entity_resolution_uses_unique_organization_match_but_not_similar_people():
    known_org = entity("Northwind Technologies", canonical="northwind technologies")
    org_mention = EntityMention("Northwind Technology", EntityType.ORGANIZATION)
    resolver = FuzzyEntityResolver(EntityStore([known_org]))
    organization = resolver.resolve(SCOPE.user_id, [org_mention])
    assert organization.entity_ids[mention_key(org_mention)] == known_org.entity_id

    known_person = entity("Individual North", EntityType.PERSON, canonical="individual north")
    person_mention = EntityMention("Individual Northe", EntityType.PERSON)
    person = FuzzyEntityResolver(EntityStore([known_person])).resolve(
        SCOPE.user_id, [person_mention]
    )
    assert person.entity_ids[mention_key(person_mention)] != known_person.entity_id


def test_entity_resolution_does_not_merge_ambiguous_or_other_user_candidates():
    candidates = [
        entity("Cedar Technologies", canonical="cedar technologies"),
        entity("Cedar Technologics", canonical="cedar technologics"),
        entity("Cedar Technology", canonical="cedar technology", user="different-user"),
    ]
    mention = EntityMention("Cedar Technolog", EntityType.ORGANIZATION)
    resolution = FuzzyEntityResolver(EntityStore(candidates)).resolve(SCOPE.user_id, [mention])
    assert resolution.entity_ids[mention_key(mention)] not in {
        candidate.entity_id for candidate in candidates
    }
    assert resolution.upserts[0].user_id == SCOPE.user_id


def test_ambiguous_exact_entity_match_falls_back_instead_of_failing():
    company = entity("Apple", EntityType.ORGANIZATION)
    product = entity("Apple", EntityType.PRODUCT)
    resolver = FuzzyEntityResolver(EntityStore([company, product]))

    typed = EntityMention("Apple", EntityType.PRODUCT)
    untyped = EntityMention("Apple", EntityType.OTHER)
    resolution = resolver.resolve(SCOPE.user_id, [typed, untyped])

    assert resolution.entity_ids[mention_key(typed)] == product.entity_id
    own = resolution.entity_ids[mention_key(untyped)]
    assert own not in {company.entity_id, product.entity_id}
    assert [e.entity_type for e in resolution.upserts] == [EntityType.OTHER]


def test_entity_lookup_sends_raw_normalized_and_suffix_free_names():
    store = EntityStore([])
    FuzzyEntityResolver(store).resolve(
        SCOPE.user_id, [EntityMention(" Google  LLC", EntityType.ORGANIZATION)]
    )
    ((_, names),) = store.queries
    assert names == ("Google  LLC", "google llc", "google")


def test_entity_resolution_collapses_exact_same_batch_mentions():
    mention_a = EntityMention("Data Platform", EntityType.CONCEPT)
    mention_b = EntityMention(" data   platform ", EntityType.CONCEPT)
    resolution = FuzzyEntityResolver(EntityStore([])).resolve(SCOPE.user_id, [mention_a, mention_b])
    assert (
        resolution.entity_ids[mention_key(mention_a)]
        == resolution.entity_ids[mention_key(mention_b)]
    )
    assert len(resolution.upserts) == 1


def test_semantic_links_keep_similar_facts_and_reject_unrelated_or_scope_leaks():
    new = fact("A system uses a vector database.", embedding=(1.0, 0.0))
    similar = fact("A system uses vector search.", fact_id="similar", embedding=(0.98, 0.2))
    unrelated = fact("A service runs batch jobs.", fact_id="unrelated", embedding=(0.0, 1.0))
    global_new = fact(
        "A global system uses vectors.",
        fact_id="global-new",
        visibility=Visibility.GLOBAL,
        embedding=(1.0, 0.0),
    )
    project_candidate = fact(
        "A project uses vectors.",
        fact_id="project-fact",
        visibility=Visibility.PROJECT,
        embedding=(1.0, 0.0),
    )
    other_project = fact(
        "Other project uses vector search.",
        fact_id="other-project",
        scope=Scope(SCOPE.user_id, "project-b", "session-b"),
        visibility=Visibility.PROJECT,
        embedding=(1.0, 0.0),
    )
    store = LinkStore([(similar, 0.98), (similar, 0.97), (unrelated, 0.1), (other_project, 0.99)])

    links = StoreLinker(store, min_similarity=0.75).link([new])

    assert [(link.to_fact_id, link.relation, link.weight) for link in links] == [
        (similar.fact_id, Relation.SIMILAR, 0.98)
    ]
    assert store.similarity_calls[0][0] == Scope(SCOPE.user_id, project_id="project-a")
    assert StoreLinker(LinkStore([(project_candidate, 1.0)])).link([global_new]) == []


def test_semantic_links_include_in_batch_pairs_once_and_skip_self_links():
    first = fact("First event", fact_id="a", embedding=(1.0, 0.0))
    second = fact("Related event", fact_id="b", embedding=(0.99, 0.1))
    links = StoreLinker(LinkStore(), min_similarity=0.8).link([first, second, first])
    assert len(links) == 1
    assert (links[0].from_fact_id, links[0].to_fact_id, links[0].relation) == (
        "a",
        "b",
        Relation.SIMILAR,
    )


def test_temporal_links_use_event_time_and_order_edges_chronologically():
    earlier = fact("Earlier event", fact_id="earlier", occurred_at=AT)
    later = fact("Later event", fact_id="later", occurred_at=AT + timedelta(hours=3))
    store = LinkStore(near=[earlier])
    linker = StoreLinker(store, temporal_window=timedelta(days=1), temporal_tau=timedelta(days=1))

    links = linker.link([later])

    assert len(links) == 1
    assert (links[0].from_fact_id, links[0].to_fact_id, links[0].relation) == (
        earlier.fact_id,
        later.fact_id,
        Relation.NEAR_IN_TIME,
    )
    assert links[0].weight == pytest.approx(0.8824969)
    assert store.temporal_calls[0][1] == later.occurred_at


def test_temporal_links_skip_unknown_dates_and_deduplicate_repeated_pairs():
    first = fact("Untimed statement", fact_id="first", occurred_at=None)
    second = fact("Also untimed", fact_id="second", occurred_at=None)
    links = StoreLinker(LinkStore(near=[first, second])).link([first, second])
    assert all(link.relation != Relation.NEAR_IN_TIME for link in links)

    earlier = fact("Earlier", fact_id="a", occurred_at=AT)
    later = fact("Later", fact_id="b", occurred_at=AT + timedelta(hours=1))
    links = StoreLinker(LinkStore(near=[earlier, later])).link([earlier, later])
    temporal = [link for link in links if link.relation == Relation.NEAR_IN_TIME]
    assert len(temporal) == 1


class ClassifierStore:
    def __init__(self, similar=(), by_entity=()):
        self.similar = list(similar)
        self.by_entity = list(by_entity)

    def similar_facts(self, *args, **kwargs):
        return self.similar

    def facts_with_entities(self, *args, **kwargs):
        return self.by_entity


def controlled_client(rows):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(rows))
    return client


def decision_row(new, action, targets=()):
    return {"fact_id": new.fact_id, "action": action, "target_ids": list(targets)}


def test_classifier_uses_relevant_candidates_and_accepts_supported_duplicate():
    old = fact("The system uses one database.", fact_id="old", category=Category.DECISION)
    new = fact("The system uses one database.", fact_id="new", category=Category.DECISION)
    unrelated_scope = fact(
        "Another project's system uses one database.",
        fact_id="project-b",
        scope=Scope(SCOPE.user_id, "project-b", "session-b"),
        visibility=Visibility.PROJECT,
        category=Category.DECISION,
    )
    client = controlled_client([decision_row(new, "duplicate", [old.fact_id])])
    store = ClassifierStore(similar=[(old, 0.99), (unrelated_scope, 1.0)])

    (decision,) = GeminiSupersedeClassifier(client, store).classify([new])

    assert decision.action == UpdateAction.DUPLICATE
    assert decision.target_ids == (old.fact_id,)
    payload = json.loads(
        client.models.generate_content.call_args.kwargs["contents"].split("INPUT:\n")[1]
    )
    candidates = payload[0]["candidates"]
    assert [candidate["fact_id"] for candidate in candidates] == [old.fact_id]
    assert (
        client.models.generate_content.call_args.kwargs["config"]["response_mime_type"]
        == "application/json"
    )


def test_classifier_does_not_offer_project_fact_as_a_candidate_to_global_fact():
    new = fact("A global preference", fact_id="global-new", visibility=Visibility.GLOBAL)
    project_old = fact("A project preference", fact_id="project-old", visibility=Visibility.PROJECT)
    client = Mock()
    store = ClassifierStore(similar=[(project_old, 0.99)])

    (decision,) = GeminiSupersedeClassifier(client, store).classify([new])

    assert decision.action == UpdateAction.ADD
    client.models.generate_content.assert_not_called()


def test_classifier_does_not_reconcile_facts_from_different_speakers():
    new = fact("A current statement", fact_id="new", category=Category.DECISION)
    assistant_fact = fact(
        "A previous statement",
        fact_id="assistant",
        category=Category.DECISION,
        claim_type=ClaimType.ASSERTED,
        speaker=Speaker.ASSISTANT,
    )
    client = Mock()

    (decision,) = GeminiSupersedeClassifier(
        client, ClassifierStore(similar=[(assistant_fact, 0.99)])
    ).classify([new])

    assert decision.action == UpdateAction.ADD
    client.models.generate_content.assert_not_called()


def test_classifier_never_supersedes_global_memory_from_a_project_fact():
    new = fact("Project setting changed", fact_id="project-new", visibility=Visibility.PROJECT)
    global_old = fact(
        "Global setting",
        fact_id="global-old",
        visibility=Visibility.GLOBAL,
        scope=Scope(SCOPE.user_id),
    )
    client = controlled_client([decision_row(new, "supersede", [global_old.fact_id])])

    (decision,) = GeminiSupersedeClassifier(
        client, ClassifierStore(similar=[(global_old, 0.99)])
    ).classify([new])

    assert decision == UpdateDecision(new.fact_id)  # kept as new; the global fact stays current


@pytest.mark.parametrize(
    "rows",
    [
        "not-json",
        {"fact_id": "new"},
        [{"fact_id": "new", "action": "unknown", "target_ids": []}],
        [{"fact_id": "new", "action": "duplicate", "target_ids": ["unlisted"]}],
        [{"fact_id": "invented", "action": "add", "target_ids": []}],
        [],
    ],
)
def test_classifier_falls_back_to_add_on_malformed_or_unsupported_output(rows):
    old = fact("A prior fact", fact_id="old")
    new = fact("A new fact", fact_id="new")
    client = Mock()
    text = rows if isinstance(rows, str) else json.dumps(rows)
    client.models.generate_content.return_value = SimpleNamespace(text=text)

    decisions = GeminiSupersedeClassifier(client, ClassifierStore(similar=[(old, 0.9)])).classify(
        [new]
    )

    assert decisions == [UpdateDecision("new")]


def test_classifier_keeps_supported_contradiction_as_new_fact():
    previous = fact("Service runs in region east", fact_id="previous", category=Category.DECISION)
    current = fact("Service runs in region west", fact_id="current", category=Category.DECISION)
    client = controlled_client([decision_row(current, "add")])

    (decision,) = GeminiSupersedeClassifier(
        client, ClassifierStore(similar=[(previous, 0.9)])
    ).classify([current])

    assert decision.action == UpdateAction.ADD
    assert (
        "Keep contradictions as separate facts"
        in client.models.generate_content.call_args.kwargs["contents"]
    )
