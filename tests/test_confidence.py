from cost_aware_router.confidence import parse_response


def test_parses_clean_json():
    assert parse_response('{"answer": "Paris", "confidence": 0.9}') == ("Paris", 0.9)


def test_extracts_json_from_surrounding_prose():
    ans, conf = parse_response('Sure! {"answer": "Paris", "confidence": 0.7} hope that helps')
    assert (ans, conf) == ("Paris", 0.7)


def test_clamps_out_of_range_confidence():
    assert parse_response('{"answer": "x", "confidence": 7}')[1] == 1.0


def test_missing_or_garbage_confidence_is_none():
    assert parse_response('{"answer": "x"}')[1] is None
    assert parse_response('{"answer": "x", "confidence": "high"}')[1] is None


def test_unparseable_text_returns_raw_text_and_no_confidence():
    assert parse_response("just prose") == ("just prose", None)
