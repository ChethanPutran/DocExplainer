"""The temporal knowledge graph: built from empty, section by section, in reading order."""
import json

import pytest

from doc_explainer.core.common.knowledge import MentionRole, RelationType
from doc_explainer.core.common.ports import LLMError, LLMFormatError
from doc_explainer.core.document import parse_document
from doc_explainer.core.knowledge import (Canonicalizer, ConceptExtractor, KnowledgeBuildError, KnowledgeView,
                                          SqliteKnowledgeStore, TemporalKnowledgeBuilder, normalize_name, plan_steps)
from doc_explainer.core.knowledge.store import delta_from_dict

from fakes import HashingEmbedder, ScriptedLLM

from fakes import BOOK, SCRIPT, c, rel  # shared with the system and API tests


class Env:
    def __init__(self, tmp_path, script=SCRIPT, book=BOOK, name="book"):
        (tmp_path / f"{name}.md").write_text(book, encoding="utf-8")
        self.structure = parse_document(tmp_path / f"{name}.md", name)
        self.llm = ScriptedLLM(script)
        self.store = SqliteKnowledgeStore(tmp_path / f"k_{name}.db")
        self.builder = TemporalKnowledgeBuilder(ConceptExtractor(self.llm), Canonicalizer(HashingEmbedder()), self.store)
        self.view = KnowledgeView(self.store)
        self.doc = self.structure.doc_id

    def names(self, upto=None):
        return {c.name for c in self.store.concepts(self.doc, upto)}

    def by_name(self, name):
        return next(c for c in self.store.concepts(self.doc) if c.name == name)


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    e.report = e.builder.build(e.structure)
    return e


# ----------------------------------------------------------------------------- the graph grows along the document
def test_the_graph_starts_empty_and_grows_one_section_at_a_time(tmp_path):
    e = Env(tmp_path)
    assert e.store.concepts(e.doc) == [] and e.store.relations(e.doc) == []
    sizes = []
    for step in plan_steps(e.structure):
        e.builder.advance(e.doc, step)
        sizes.append(len(e.store.concepts(e.doc)))
    assert sizes == [2, 4, 7, 8]                                   # +photosynthesis,chlorophyll / +light reactions,ATP / +3 / +1


def test_a_full_build_finds_each_concept_once_with_its_place_in_the_text(env):
    assert env.names() == {"photosynthesis", "chlorophyll", "light reactions", "ATP", "Calvin cycle", "carbon dioxide",
                           "sugar", "cellular respiration"}
    assert env.report.steps_done == 4 and env.report.steps_failed == 0 and env.report.concepts == 8
    atp = env.by_name("ATP")
    assert atp.introduced_ordinal == env.structure.node("s1.1/p1").ordinal            # where it first appeared
    assert atp.first_node_id == "s1.1/p1" and atp.defined_ordinal == atp.introduced_ordinal
    introduced = [c.introduced_ordinal for c in env.store.concepts(env.doc)]
    assert introduced == sorted(introduced)                                            # concepts are in reading order


def test_variants_of_a_name_are_merged_not_duplicated(env):
    # "atp", "Light Reactions" and "sugars" in later sections are the SAME concepts as the earlier "ATP", "light reactions", "sugar"
    for name in ("ATP", "light reactions", "sugar"):
        assert sum(1 for c in env.store.concepts(env.doc) if normalize_name(c.name) == normalize_name(name)) == 1
    atp = env.by_name("ATP")
    nodes = {m.node_id for m in env.store.mentions(env.doc, atp.id)}
    assert nodes == {"s1.1/p1", "s1.2/p1"} and atp.mention_count == 2                  # mentioned in two sections
    assert env.by_name("sugar").aliases == []                                           # plural/case variants are not new aliases


def test_known_ids_given_by_the_model_are_honoured(env):
    chlorophyll = env.by_name("chlorophyll")
    assert chlorophyll.id == "c2" and {m.node_id for m in env.store.mentions(env.doc, "c2")} == {"s1/p2", "s1.1/p1"}


# ----------------------------------------------------------------------------- nothing may depend on the future
def test_prerequisites_only_point_from_earlier_to_later_and_never_loop(env):
    concepts = {c.id: c for c in env.store.concepts(env.doc)}
    prereq = env.store.relations(env.doc, type=RelationType.PREREQUISITE_OF)
    pairs = {(concepts[r.src].name, concepts[r.dst].name) for r in prereq}
    assert pairs == {("chlorophyll", "light reactions"), ("light reactions", "Calvin cycle"),
                     ("carbon dioxide", "sugar"), ("sugar", "cellular respiration")}
    for r in prereq:
        assert concepts[r.src].introduced_ordinal <= concepts[r.dst].introduced_ordinal
    adj = {}
    for r in prereq:
        adj.setdefault(r.src, set()).add(r.dst)

    def cyclic(n, path):
        return n in path or any(cyclic(m, path | {n}) for m in adj.get(n, ()))
    assert not any(cyclic(n, frozenset()) for n in adj)


def test_what_the_model_got_wrong_is_rejected_with_a_reason(env):
    reasons = [r for d in env.store.deltas(env.doc) for r in d["rejected"]]
    assert any(r.startswith("forward dependency") and "'chlorophyll' appears after 'photosynthesis'" in r for r in reasons)
    assert any(r.startswith("forward dependency") and "'Calvin cycle' appears after 'light reactions'" in r for r in reasons)
    assert any("unknown concept" in r and "ghost" in r for r in reasons)


def test_a_prerequisite_cycle_among_things_introduced_together_is_rejected(tmp_path):
    # Concepts met in the same paragraph have no reading order between them, so only the cycle check can protect the graph.
    book = "# Loops\n\nAlpha, beta and gamma are introduced together here.\n"
    script = {"Loops": {"concepts": [c("alpha"), c("beta"), c("gamma")],
                        "relations": [rel("alpha", "beta", "prerequisite_of"), rel("beta", "gamma", "prerequisite_of"),
                                      rel("gamma", "alpha", "prerequisite_of"), rel("gamma", "beta", "prerequisite_of")]}}
    e = Env(tmp_path, script=script, book=book)
    e.builder.build(e.structure)
    names = {c.id: c.name for c in e.store.concepts(e.doc)}
    kept = {(names[r.src], names[r.dst]) for r in e.store.relations(e.doc, type=RelationType.PREREQUISITE_OF)}
    assert kept == {("alpha", "beta"), ("beta", "gamma")}
    reasons = [r for d in e.store.deltas(e.doc) for r in d["rejected"]]
    assert len([r for r in reasons if r.startswith("cycle")]) == 2


def test_repeated_support_strengthens_a_relation_instead_of_duplicating_it(tmp_path):
    script = {"Photosynthesis": {"concepts": [c("photosynthesis", paragraphs=(1,)), c("chlorophyll", paragraphs=(1, 2))],
                                 "relations": [rel("chlorophyll", "photosynthesis", "part_of", 1), rel("chlorophyll", "photosynthesis", "part_of", 2)]},
              "Light reactions": {"concepts": [c("chlorophyll"), c("photosynthesis")],
                                  "relations": [rel("chlorophyll", "photosynthesis", "part_of")]}}
    e = Env(tmp_path, script=script)
    e.builder.build(e.structure)
    parts = [r for r in e.store.relations(e.doc) if r.type is RelationType.PART_OF]
    assert len(parts) == 1 and parts[0].strength == 3 and len(parts[0].evidence) >= 2


def test_symmetric_relations_are_stored_once_earliest_concept_first(env):
    rel_ = next(r for r in env.store.relations(env.doc) if r.type is RelationType.CONTRASTS_WITH)
    assert (env.by_name("photosynthesis").id, env.by_name("cellular respiration").id) == (rel_.src, rel_.dst)


def test_building_a_step_never_shows_the_model_text_from_later_sections(env):
    for i, prompt in enumerate(env.llm.prompts):
        later = [t for t in ("Calvin cycle uses ATP", "Cellular respiration releases") if t in prompt and i < {"Calvin cycle uses ATP": 2, "Cellular respiration releases": 3}[t]]
        assert not later, f"step {i} saw future text: {later}"
    assert "(none yet" in env.llm.prompts[0]                                          # step 0 starts from nothing
    assert "c1 | photosynthesis | Making chemical energy from light." in env.llm.prompts[1]
    assert "light reactions" in env.llm.prompts[2] and "ATP" in env.llm.prompts[2]


# ----------------------------------------------------------------------------- the graph as of a point in the document
def test_the_graph_as_of_a_position_is_what_a_reader_there_has_met(env):
    s = env.structure
    after_light = env.view.graph_at(env.doc, s.own_end("s1.1"))
    assert {x.name for x in after_light.concepts} == {"photosynthesis", "chlorophyll", "light reactions", "ATP"}
    ids = {x.id for x in after_light.concepts}
    assert all(r.src in ids and r.dst in ids for r in after_light.relations)
    assert env.view.graph_at(env.doc, 0).concepts == []
    assert len(env.view.graph_at(env.doc).concepts) == 8
    assert [len(env.view.graph_at(env.doc, s.own_end(x)).concepts) for x in ("s1", "s1.1", "s1.2", "s1.3")] == [2, 4, 7, 8]


def test_state_at_any_point_equals_replaying_the_delta_history_up_to_it(env):
    deltas = env.store.deltas(env.doc)
    for sec in env.structure.sections():
        t = env.structure.own_end(sec.id)
        replay = {c["id"] for d in deltas if d["ordinal_start"] <= t for c in d["new_concepts"] if c["introduced_ordinal"] <= t}
        assert replay == {c.id for c in env.view.graph_at(env.doc, t).concepts}


def test_definitions_are_remembered_as_the_text_evolves(env):
    p = env.by_name("photosynthesis")
    assert p.definition == "Making chemical energy from light."                         # the first definition stays primary
    early = env.structure.own_end("s1.1")
    late = env.structure.own_end("s1.3")
    assert env.view.glossary(env.doc, early, [p.id]) == {p.id: "Making chemical energy from light."}
    assert env.view.glossary(env.doc, late, [p.id])[p.id].startswith("The opposite of respiration")
    assert len(env.store.definitions(env.doc, p.id)) == 2


# ----------------------------------------------------------------------------- what each section needs and teaches
def test_each_section_introduces_some_concepts_and_relies_on_earlier_ones(env):
    k = env.view.section_knowledge(env.doc, env.structure, "s1.2")                     # Calvin cycle
    assert {x.name for x in k.introduces} == {"Calvin cycle", "carbon dioxide", "sugar"}
    assert {x.name for x in k.uses_earlier} == {"ATP", "light reactions"}
    assert {x.name for x in env.view.section_knowledge(env.doc, env.structure, "s1").introduces} == {"photosynthesis", "chlorophyll"}
    assert env.view.section_knowledge(env.doc, env.structure, "s1").uses_earlier == []   # the first section needs nothing


def test_sections_depend_only_on_earlier_sections(env):
    deps = env.view.section_dependencies(env.doc, env.structure)
    calvin = {d.section_id: d for d in deps["s1.2"]}
    assert set(calvin) == {"s1.1"} and calvin["s1.1"].weight == 2                        # via ATP and light reactions
    respiration = {d.section_id for d in deps["s1.3"]}
    assert respiration == {"s1", "s1.2"}                                                  # photosynthesis, sugar
    assert deps["s1"] == []
    for sid, ds in deps.items():
        for d in ds:
            assert env.structure.node(d.section_id).ordinal < env.structure.node(sid).ordinal   # never forward


def test_the_knowledge_growth_curve(env):
    g = env.view.growth(env.doc)
    assert [p.total_concepts for p in g] == [2, 4, 7, 8]
    assert [p.new_concepts for p in g] == [2, 2, 3, 1]
    assert g[0].reuse == 0.0 and 0 < g[2].reuse < 1                                      # later sections reuse earlier concepts
    assert all(p.novelty > 0 for p in g) and [p.total_relations for p in g] == sorted(p.total_relations for p in g)


# ----------------------------------------------------------------------------- dependencies between concepts
def test_prerequisites_and_learning_path(env):
    target = env.by_name("cellular respiration")
    chain = env.view.prerequisites(env.doc, target.id, depth=3)
    assert [(p.concept.name, p.distance) for p in chain] == [("sugar", 1), ("carbon dioxide", 2)]
    path = env.view.learning_path(env.doc, target.id)
    assert [x.name for x in path] == ["carbon dioxide", "sugar", "cellular respiration"]
    assert [x.introduced_ordinal for x in path] == sorted(x.introduced_ordinal for x in path)   # the order the book teaches it
    assert [p.concept.name for p in env.view.prerequisites(env.doc, target.id, depth=1)] == ["sugar"]
    assert env.view.learning_path(env.doc, "nope") == []
    # as of an earlier position the later prerequisites are not yet known
    assert env.view.prerequisites(env.doc, env.by_name("Calvin cycle").id, upto=env.structure.own_end("s1.1")) == []


def test_related_concepts_assumed_background_and_name_lookup(env):
    rel_names = {x.name for x, _ in env.view.related(env.doc, env.by_name("photosynthesis").id)}
    assert {"chlorophyll", "cellular respiration"} <= rel_names
    assert {x.name for x in env.view.assumed(env.doc)} == {"carbon dioxide", "sugar"}       # used, never defined in the text
    assert env.view.find(env.doc, "The Photosynthesis").name == "photosynthesis"
    assert env.view.find(env.doc, "sugars").name == "sugar" and env.view.find(env.doc, "unobtainium") is None


# ----------------------------------------------------------------------------- failure and resume
def test_a_failed_step_is_recorded_and_retried_in_order_on_the_next_run(tmp_path):
    e = Env(tmp_path)
    e.llm.fail["Calvin cycle"] = LLMError("quota exceeded")
    first = e.builder.build(e.structure)
    assert (first.steps_done, first.steps_failed) == (3, 1) and "quota exceeded" in first.errors[0]
    failed = [r for r in e.store.steps(e.doc) if r.status == "failed"]
    assert len(failed) == 1 and failed[0].section_id == "s1.2" and "quota exceeded" in failed[0].error
    calls_before = len(e.llm.prompts)
    del e.llm.fail["Calvin cycle"]
    second = e.builder.build(e.structure)
    assert second.steps_failed == 0 and second.steps_skipped == 2                          # sections 0 and 1 were kept
    assert len(e.llm.prompts) - calls_before == 2                                          # Calvin cycle + the one after it, redone
    (tmp_path / "ref").mkdir()
    reference = Env(tmp_path / "ref")                                                      # a clean run for comparison
    reference.builder.build(reference.structure)
    assert {(c.name, c.introduced_ordinal) for c in e.store.concepts(e.doc)} == {(c.name, c.introduced_ordinal) for c in reference.store.concepts(reference.doc)}
    assert e.names() == reference.names()


def test_a_finished_build_is_not_redone_and_a_changed_document_starts_over(tmp_path):
    e = Env(tmp_path)
    e.builder.build(e.structure)
    n = len(e.llm.prompts)
    again = e.builder.build(e.structure)
    assert len(e.llm.prompts) == n and again.steps_skipped == 4 and again.steps_done == 0
    changed = parse_document(_write(tmp_path, "book.md", BOOK.replace("green pigment", "green dye")), "book")
    e.builder.build(changed)
    assert len(e.llm.prompts) == n + 4                                                     # edited text: everything redone


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path / name


def test_the_build_stops_early_and_loudly_when_the_model_keeps_failing(tmp_path):
    e = Env(tmp_path, script={t: LLMError("401 invalid api key") for t in ("Photosynthesis", "Light reactions", "Calvin cycle", "Cellular respiration")})
    with pytest.raises(KnowledgeBuildError, match="401 invalid api key"):
        e.builder.build(e.structure)
    assert len(e.llm.prompts) == 3                                                          # gave up after 3, not all 4
    assert e.store.concepts(e.doc) == []                                                    # and left no half-empty "graph"


def test_mostly_failing_builds_fail_even_if_not_consecutively(tmp_path):
    e = Env(tmp_path)
    e.llm.fail.update({"Photosynthesis": LLMError("x"), "Calvin cycle": LLMError("y")})
    with pytest.raises(KnowledgeBuildError, match="2 of 4 steps failed"):
        e.builder.build(e.structure)


def test_cancelling_stops_between_steps(tmp_path):
    e = Env(tmp_path)
    count = {"n": 0}

    def cancelled():
        count["n"] += 1
        return count["n"] > 2
    report = e.builder.build(e.structure, cancelled=cancelled)
    assert report.steps_done == 2 and len(e.llm.prompts) == 2
    resumed = e.builder.build(e.structure)
    assert resumed.steps_skipped == 2 and resumed.steps_done == 2


def test_progress_is_reported_in_order(tmp_path):
    e = Env(tmp_path)
    seen = []
    e.builder.build(e.structure, progress=lambda stage, f: seen.append((stage, f)))
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and seen[0][0].startswith("reading:")


def test_rewind_restores_exactly_the_earlier_state(tmp_path):
    e = Env(tmp_path)
    e.builder.build(e.structure)
    (tmp_path / "x").mkdir()
    reference = Env(tmp_path / "x", name="ref")
    for step in plan_steps(reference.structure)[:2]:
        reference.builder.advance(reference.doc, step)
    e.store.rewind(e.doc, 2)
    state = lambda env_: ([(c.id, c.name, c.introduced_ordinal, c.importance, c.mention_count, c.definition) for c in env_.store.concepts(env_.doc)],
                          [(r.key, r.strength) for r in env_.store.relations(env_.doc)])
    assert state(e) == state(reference)
    assert [r.step for r in e.store.steps(e.doc)] == [0, 1] and len(e.store.deltas(e.doc)) == 2


def test_deltas_survive_a_json_round_trip(env):
    for raw in env.store.deltas(env.doc):
        d = delta_from_dict(json.loads(json.dumps(raw)))
        assert d.step == raw["step"] and len(d.new_concepts) == len(raw["new_concepts"])
        assert all(isinstance(m.role, MentionRole) for m in d.mentions)
        assert all(isinstance(r.type, RelationType) for r in d.new_relations)


# ----------------------------------------------------------------------------- tolerance of messy model output
def test_malformed_output_is_cleaned_not_trusted(tmp_path):
    messy = {"concepts": [
        "not a dict", {"name": 5}, {"name": ""}, {"name": "x"}, {"name": "figure"}, {"name": "42"},
        {"name": "  Mitosis  ", "role": "NONSENSE", "paragraphs": ["P1", 99, -3, None, 1.0], "importance": 99, "known_id": 7,
         "aliases": ["cell division", "", 3], "definition": "x" * 500},
        {"name": "mitosis", "role": "defines", "paragraphs": [1], "definition": "Cells divide.", "importance": 1},
        {"name": "meiosis", "paragraphs": "garbage"}],
        "relations": ["junk", {"source": "mitosis"}, {"source": "mitosis", "target": "meiosis", "type": "Contrasts-With", "paragraph": 99},
                      {"source": "mitosis", "target": "meiosis", "type": "invented_type"}],
        "extra_field": [1, 2, 3]}
    ex = ConceptExtractor(ScriptedLLM({})).parse(messy, n_paragraphs=1)
    by = {c.name.lower(): c for c in ex.concepts}
    assert set(by) == {"mitosis", "meiosis"}                                                # junk and stop-words dropped, dupes merged
    m = by["mitosis"]
    assert m.paragraphs == [1] and m.importance == 3 and m.known_id is None and m.aliases == ["cell division"]
    assert m.role is MentionRole.DEFINES and len(m.definition) == 300                        # first definition kept, capped
    assert by["meiosis"].paragraphs == [1]                                                  # bad paragraph info -> first paragraph
    assert [(r.source, r.target, r.type, r.paragraph) for r in ex.relations] == [("mitosis", "meiosis", RelationType.CONTRASTS_WITH, None)]
    assert ConceptExtractor(ScriptedLLM({})).parse([], 3).concepts == [] and ConceptExtractor(ScriptedLLM({})).parse(None, 3).relations == []


def test_the_extractor_does_not_repeat_the_same_request_on_bad_json(tmp_path):
    """Retrying an empty/cut-off answer is the LLM client's job and it uses a bigger budget (test_token_escalation.py);
    the extractor must not send the same prompt again on top of that."""
    calls = []

    class AlwaysBad:
        def ask_json(self, system, user):
            calls.append(user)
            raise LLMFormatError("still not json")
    e = Env(tmp_path)
    with pytest.raises(LLMFormatError):
        ConceptExtractor(AlwaysBad()).extract(plan_steps(e.structure)[0], [])
    assert len(calls) == 1


# ----------------------------------------------------------------------------- steps and names
def test_steps_are_sections_in_order_with_only_their_own_text(tmp_path):
    e = Env(tmp_path)
    steps = plan_steps(e.structure)
    assert [s.section_id for s in steps] == ["s1", "s1.1", "s1.2", "s1.3"] and [s.index for s in steps] == [0, 1, 2, 3]
    assert [len(s.paragraphs) for s in steps] == [2, 1, 1, 1]
    assert "Light reactions" not in " ".join(t for _, _, t in steps[0].paragraphs)          # no subsection text in the parent's step
    assert steps[1].ordinal_start == e.structure.node("s1.1/p1").ordinal
    assert plan_steps(e.structure) == plan_steps(e.structure)                               # deterministic


def test_long_sections_are_windowed_and_empty_ones_skipped(tmp_path):
    body = "\n\n".join(f"Paragraph {i} " + "word " * 100 for i in range(10))
    (tmp_path / "l.md").write_text(f"# Long\n\n{body}\n\n## Empty\n\n## Next\n\nShort.\n", encoding="utf-8")
    steps = plan_steps(parse_document(tmp_path / "l.md", "l"), max_words=450)
    assert len([s for s in steps if s.section_id == "s1"]) >= 3 and all(s.words <= 450 + 110 for s in steps)
    assert "s1.1" not in {s.section_id for s in steps} and steps[-1].title == "Next"
    assert [s.index for s in steps] == list(range(len(steps)))


@pytest.mark.parametrize("a,b", [("Photosynthesis", "photosynthesis"), ("the Calvin cycle", "Calvin Cycle"), ("sugars", "sugar"),
                                 ("Calvin cycle (C3)", "calvin cycle"), ("enzymes", "enzyme"), ("theories", "theory")])
def test_names_that_should_match_do(a, b):
    assert normalize_name(a) == normalize_name(b)


@pytest.mark.parametrize("a,b", [("mitosis", "meiosis"), ("ATP", "ADP"), ("gas", "glass")])
def test_names_that_must_stay_apart_do(a, b):
    assert normalize_name(a) != normalize_name(b)


def test_embedding_similarity_merges_only_near_identical_names():
    from doc_explainer.core.common.knowledge import Concept
    canon = Canonicalizer(HashingEmbedder(), threshold=0.9)
    canon.add(Concept("c1", "light dependent reactions"))
    canon.add(Concept("c2", "mitosis"))
    assert canon.match(["dependent light reactions"]) == "c1"                               # same words: close enough
    assert canon.match(["meiosis"]) is None                                                  # different: stays separate
    assert canon.match(["Mitosis"]) == "c2"


def test_snake_case_concept_names_become_words_but_maths_symbols_are_kept():
    from doc_explainer.core.knowledge.extractor import _clean_name
    assert _clean_name("fourier_neural_operator") == "fourier neural operator"
    assert _clean_name("weight matrix w_ij") == "weight matrix w_ij"
    assert _clean_name("w_ij") == "w_ij"
