import importlib
import sys, types
import pytest

@pytest.fixture(autouse=True)
def add_module_stubs(monkeypatch):
    # stub heavy modules used by think.indexer
    if 'usearch.index' not in sys.modules:
        usearch = types.ModuleType('usearch')
        index_mod = types.ModuleType('usearch.index')
        class DummyIndex:
            def __init__(self, *a, **k):
                pass
            def save(self, *a, **k):
                pass
            @classmethod
            def restore(cls, *a, **k):
                return cls()
            def remove(self, *a, **k):
                pass
            def add(self, *a, **k):
                pass
            def search(self, *a, **k):
                class Res: keys=[1]; distances=[0.0]
                return Res()
        index_mod.Index = DummyIndex
        usearch.index = index_mod
        sys.modules['usearch'] = usearch
        sys.modules['usearch.index'] = index_mod
    if 'sentence_transformers' not in sys.modules:
        st_mod = types.ModuleType('sentence_transformers')
        class DummyST:
            def __init__(self, *a, **k):
                pass
            def get_sentence_embedding_dimension(self):
                return 384
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [([0.0]*384) for _ in texts]
        st_mod.SentenceTransformer = DummyST
        sys.modules['sentence_transformers'] = st_mod
    if 'sklearn.metrics.pairwise' not in sys.modules:
        pairwise = types.ModuleType('pairwise')
        def cosine_similarity(a, b):
            return [[1.0]]
        pairwise.cosine_similarity = cosine_similarity
        metrics = types.ModuleType('metrics')
        metrics.pairwise = pairwise
        sys.modules['sklearn'] = types.ModuleType('sklearn')
        sys.modules['sklearn.metrics'] = metrics
        sys.modules['sklearn.metrics.pairwise'] = pairwise
    if 'dotenv' not in sys.modules:
        dotenv_mod = types.ModuleType('dotenv')
        def load_dotenv(*a, **k):
            return True
        dotenv_mod.load_dotenv = load_dotenv
        sys.modules['dotenv'] = dotenv_mod
    if 'input_detect' not in sys.modules:
        input_detect_mod = types.ModuleType('input_detect')
        def input_detect():
            return None, None
        input_detect_mod.input_detect = input_detect
        sys.modules['input_detect'] = input_detect_mod
        sys.modules['hear.input_detect'] = input_detect_mod
    if 'gi' not in sys.modules:
        gi_mod = types.ModuleType('gi')
        gi_mod.require_version = lambda *a, **k: None
        class Dummy(types.ModuleType):
            pass
        repo = types.ModuleType('gi.repository')
        repo.Gdk = Dummy('Gdk')
        repo.Gtk = Dummy('Gtk')
        gi_mod.repository = repo
        sys.modules['gi'] = gi_mod
        sys.modules['gi.repository'] = repo
        sys.modules['Gdk'] = repo.Gdk
        sys.modules['Gtk'] = repo.Gtk
    if 'see.screen_dbus' not in sys.modules:
        screen_dbus = types.ModuleType('see.screen_dbus')
        screen_dbus.screen_snap = lambda: []
        screen_dbus.idle_time_ms = lambda: 0
        sys.modules['see.screen_dbus'] = screen_dbus
        sys.modules['screen_dbus'] = screen_dbus
    if 'screen_compare' not in sys.modules:
        mod = importlib.import_module('see.screen_compare')
        sys.modules['screen_compare'] = mod
    if 'nltk' not in sys.modules:
        nltk_mod = types.ModuleType('nltk')
        nltk_mod.download = lambda *a, **k: True
        nltk_mod.sent_tokenize = lambda t: t.split('.')
        sys.modules['nltk'] = nltk_mod
    monkeypatch.setattr('nltk.download', lambda *a, **k: True, raising=False)
    monkeypatch.setattr('nltk.download', lambda *a, **k: True, raising=False)
