from superagent_playground.demo import DEMO_QUESTION, ScriptedLLM, run_offline_demo


def test_offline_demo_completes_through_the_react_loop():
    result = run_offline_demo(verbose=False)

    assert result.ok
    assert result.reason == "finish"
    assert "60" in result.answer
    assert len(result.steps) == 2
    assert result.steps[0].action == "Calculator[(12 + 8) * 3]"
    assert result.steps[0].observation == "60"


def test_scripted_llm_replays_responses():
    llm = ScriptedLLM(["first", "second"])

    assert llm.think([{"role": "user", "content": "x"}]) == "first"
    assert llm.think([{"role": "user", "content": "y"}]) == "second"
    assert llm.think([{"role": "user", "content": "z"}]) == "second"
    assert len(llm.calls) == 3


def test_demo_question_is_non_empty():
    assert DEMO_QUESTION.strip()
