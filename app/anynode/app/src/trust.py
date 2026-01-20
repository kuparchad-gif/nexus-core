from .schemas import VerificationResult
from datetime import datetime

class TrustScorer:
    def __init__(self, source_reliability_priors: dict = None):
        if source_reliability_priors is None:
            self.source_reliability_priors = {}
        else:
            self.source_reliability_priors = source_reliability_priors

    def calculate_score(self, verification_result: VerificationResult, recency: datetime) -> float:
        trust_score = verification_result.consensus_score

        if verification_result.entailment_scores:
            avg_entailment = sum(verification_result.entailment_scores.values()) / len(verification_result.entailment_scores)
            trust_score = (trust_score + avg_entailment) / 2

        if self.source_reliability_priors:
            source_scores = []
            for source in verification_result.entailment_scores.keys():
                if source in self.source_reliability_priors:
                    source_scores.append(self.source_reliability_priors[source])

            if source_scores:
                avg_source_score = sum(source_scores) / len(source_scores)
                trust_score = (trust_score + avg_source_score) / 2

        days_old = (datetime.now() - recency).days
        recency_score = max(0, 1 - (days_old / 365))
        trust_score = (trust_score + recency_score) / 2

        return trust_score
