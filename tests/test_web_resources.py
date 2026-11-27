"""Internet resources are search links built by strategies; nothing is invented and nothing is fetched."""
from urllib.parse import parse_qs, urlparse

from doc_explainer.core.common.enums import ExplanationLevel, ResourceType
from doc_explainer.core.tutor.splitter import FollowUpSplitter
from doc_explainer.core.tutor.web_resources import (DEFAULT_STRATEGIES, ArticleStrategy, ResourceStrategy, VideoStrategy,
                                                    web_resources)

B, A, E = ExplanationLevel.BEGINNER, ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT


def test_each_kind_builds_a_working_search_link_for_the_concept():
    out = {r.source: r for r in web_resources(["Fourier transform"], B, per_concept=9)}
    assert set(out) >= {"YouTube", "Wikipedia", "Khan Academy", "Google"}
    yt = urlparse(out["YouTube"].url)
    assert yt.netloc == "www.youtube.com" and "Fourier transform" in parse_qs(yt.query)["search_query"][0]
    assert "for beginners" in parse_qs(yt.query)["search_query"][0]
    assert out["Wikipedia"].url.startswith("https://en.wikipedia.org/w/index.php?search=Fourier+transform")
    assert out["Khan Academy"].type is ResourceType.COURSE and out["YouTube"].type is ResourceType.VIDEO


def test_depth_matches_the_reader_papers_for_advanced_tutorials_for_others():
    assert ArticleStrategy().build("FNO", A).source == "Google Scholar" and ArticleStrategy().build("FNO", E).source == "Google Scholar"
    assert ArticleStrategy().build("FNO", B).source == "Google" and "tutorial" in ArticleStrategy().build("FNO", B).url


def test_special_characters_are_escaped_and_blank_concepts_are_skipped():
    r = web_resources(["C++ & R", "  ", ""], B, per_concept=1)
    assert len(r) == 1 and "C%2B%2B+%26+R" in r[0].url


def test_limits_kinds_and_custom_strategies():
    assert len(web_resources(["a", "b"], B, per_concept=2)) == 4
    only = web_resources(["a"], B, kinds=[ResourceType.VIDEO])
    assert [r.type for r in only] == [ResourceType.VIDEO]

    class Forum(ResourceStrategy):
        kind, source = ResourceType.DOCUMENTATION, "Forum"
        def build(self, concept, level, context=""):
            return VideoStrategy().build(concept, level)
    assert web_resources(["a"], B, strategies=[Forum()])[0].title.startswith("Video")
    assert len(DEFAULT_STRATEGIES) >= 5


# ----------------------------------------------------------------------------- the streamed reply splitter
MARK = "===FOLLOW-UPS==="


def run(pieces):
    s, out = FollowUpSplitter(MARK), []
    for p in pieces:
        out += s.feed(p)
    out += s.finish_text()
    return "".join(out), s.result()


def test_the_marker_never_reaches_the_reader_however_it_is_chunked():
    text = f"The answer is here.\n{MARK}\n- Why?\n- How?\n"
    for size in (1, 2, 3, 7, len(text)):
        shown, (body, qs) = run([text[i:i + size] for i in range(0, len(text), size)])
        assert shown == "The answer is here." == body and MARK not in shown and "=" not in shown
        assert qs == ["Why?", "How?"]


def test_text_is_released_as_it_arrives_not_all_at_the_end():
    s = FollowUpSplitter(MARK)
    first = s.feed("A fairly long first sentence that is certainly not a marker. ")
    assert first and "".join(first).startswith("A fairly long")


def test_no_marker_means_no_follow_ups_and_all_the_text():
    shown, (body, qs) = run(["Just ", "an answer", " with no questions."])
    assert shown == body == "Just an answer with no questions." and qs == []


def test_numbered_and_bulleted_questions_are_cleaned_and_capped_at_three():
    _, (_, qs) = run([f"x\n{MARK}\n1. What?\n* Why?\n• How?\n- Extra?\n"])
    assert qs == ["What?", "Why?", "How?"]
