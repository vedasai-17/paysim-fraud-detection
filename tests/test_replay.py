from threading import Event

import httpx
import pandas as pd

from simulator import replay as simulator
from training.prepare import profile


def test_held_out_export_and_stoppable_replay(tmp_path, monkeypatch):
    source = 'data/fixture/paysim_sample.csv'
    target = tmp_path / 'test.csv'
    data_profile = profile(source)
    assert simulator.export_test_events(source, data_profile, target) == 6
    events = pd.read_csv(target)
    assert events.step.tolist() == [9, 9, 9, 10, 10, 10]
    assert events.isFraud.sum() == 2

    posted = []

    def response(request):
        posted.append(request.content)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(response))
    monkeypatch.setattr(simulator.httpx, 'Client', lambda **kwargs: client)
    stop = Event()
    sent = simulator.replay(target, 'http://api', 1, limit=6, run_id='new-run', stop=stop,
                            on_event=lambda *_: stop.set())
    assert sent == 1
    assert len(posted) == 1 and b'new-run-replay-0-9' in posted[0]
