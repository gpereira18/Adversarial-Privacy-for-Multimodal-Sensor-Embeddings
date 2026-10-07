from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


@dataclass
class PIISpan:
    start: int
    end: int
    entity_type: str
    score: float


def _build_engine() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine())


class PIIDetector:
    def __init__(self):
        self._engine = _build_engine()

    def detect(self, text: str) -> list[PIISpan]:
        results = self._engine.analyze(text=text, language="en")
        results = sorted(results, key=lambda r: r.start)
        return [PIISpan(r.start, r.end, r.entity_type, r.score) for r in results]

    def mask(self, text: str) -> tuple[str, list[PIISpan]]:
        spans = self.detect(text)
        masked = []
        cursor = 0
        for span in spans:
            masked.append(text[cursor:span.start])
            masked.append(f"[{span.entity_type}]")
            cursor = span.end
        masked.append(text[cursor:])
        return "".join(masked), spans

    def binary_label(self, text: str) -> int:
        return int(len(self.detect(text)) > 0)


if __name__ == "__main__":
    sample = (
        "Hi, my name is John Smith. You can reach me at john.smith@email.com "
        "or call 555-867-5309. I live at 42 Wallaby Way, Sydney."
    )
    detector = PIIDetector()
    masked, spans = detector.mask(sample)

    print("Original :", sample)
    print("Masked   :", masked)
    print("Label    :", detector.binary_label(sample), "(1 = PII detected)")
    print("\nDetected spans:")
    for s in spans:
        print(f"  [{s.start}:{s.end}] {s.entity_type:25s} (confidence {s.score:.2f})  → '{sample[s.start:s.end]}'")
