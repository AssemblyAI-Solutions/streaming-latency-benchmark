from typing import List, Tuple

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from latency_benchmark.models import (
    AudioChunkProcessing,
    BenchmarkResult,
    MatchedTranscript,
    RunOutput,
    StreamingTranscript,
    TranscribedWord,
)


class LatencyBenchmarker:
    """Computes per-word emission latency by aligning ground truth with streamed transcripts."""

    def __init__(self):
        self._normalizer = EnglishTextNormalizer()

    def _normalize(self, text: str) -> str:
        return self._normalizer(text) if text else ""

    def _get_final_text(self, transcripts: List[StreamingTranscript]) -> str:
        """Concatenate all final transcript texts."""
        return " ".join(t.text for t in transcripts if t.is_final)

    def _get_aligned_words(
        self, reference_text: str, streamed_text: str
    ) -> Tuple[List[str], float]:
        """Find words present in both reference and streamed text via edit-distance alignment."""
        ref_norm = self._normalize(reference_text)
        hyp_norm = self._normalize(streamed_text)

        if not ref_norm or not hyp_norm:
            return [], 1.0

        out = jiwer.process_words(ref_norm, hyp_norm)
        aligned = []
        for alignment in out.alignments[0]:
            if alignment.type == "equal":
                aligned.extend(out.references[0][alignment.ref_start_idx : alignment.ref_end_idx])

        return aligned, out.wer

    def _match_words_to_reference(
        self, aligned_words: List[str], ground_truth: List[TranscribedWord]
    ) -> List[TranscribedWord]:
        """Map each aligned word back to its ground-truth timestamp."""
        remaining = list(aligned_words)
        matched = []
        previous_unmatched = []

        for ref_word in ground_truth:
            if not remaining:
                break

            words_to_check = self._normalize(
                " ".join(previous_unmatched + [ref_word.text])
            ).split()

            last_word_matched = False
            for i, w in enumerate(words_to_check):
                if remaining and w == remaining[0]:
                    matched.append(
                        TranscribedWord(text=w, start_ms=ref_word.start_ms, end_ms=ref_word.end_ms)
                    )
                    remaining.pop(0)
                    if i == len(words_to_check) - 1:
                        last_word_matched = True

            if last_word_matched:
                previous_unmatched = []
            else:
                previous_unmatched.append(ref_word.text)

        return matched

    def _match_words_to_transcripts(
        self, aligned_words: List[str], transcripts: List[StreamingTranscript]
    ) -> List[MatchedTranscript]:
        """For each aligned word, find the first transcript containing it."""
        remaining = list(aligned_words)
        matched = []
        lookup_start = 0
        prev_unmatched_finals = []

        for transcript in transcripts:
            if not remaining:
                break

            last_matched = False
            words_to_check = self._normalize(
                " ".join(prev_unmatched_finals + [transcript.text])
            ).split()[lookup_start:]

            for i, w in enumerate(words_to_check):
                if remaining and w == remaining[0]:
                    remaining.pop(0)
                    last_matched = i == len(words_to_check) - 1
                    matched.append(
                        MatchedTranscript(
                            transcript=transcript,
                            is_last_word_of_final=last_matched,
                        )
                    )
                    lookup_start += 1

            if transcript.is_final and last_matched:
                lookup_start = 0
                prev_unmatched_finals = [w.text for w in transcript.words[-2:]]
            elif transcript.is_final:
                prev_unmatched_finals.append(transcript.text)

        return matched

    def _compute_latencies(
        self,
        chunks: List[AudioChunkProcessing],
        ref_words: List[TranscribedWord],
        matched_transcripts: List[MatchedTranscript],
    ) -> List[int]:
        """Compute per-word emission latency in milliseconds."""
        latencies = []
        chunk_idx = 0

        for ref_word, matched in zip(ref_words, matched_transcripts):
            while (
                chunk_idx < len(chunks) - 1
                and chunks[chunk_idx].audio_end_ts_ms < ref_word.end_ms
            ):
                chunk_idx += 1

            if chunks[chunk_idx].audio_end_ts_ms < ref_word.end_ms:
                continue

            latency_ms = round(
                (matched.transcript.abs_processing_ts - chunks[chunk_idx].processing_ts) * 1000
            )
            latencies.append(latency_ms)

        return latencies

    def run(
        self, output: RunOutput, ground_truth: List[TranscribedWord]
    ) -> BenchmarkResult:
        """Run the full benchmarking pipeline for a single audio file."""
        ref_text = " ".join(w.text for w in ground_truth)
        streamed_text = self._get_final_text(output.transcripts)

        aligned_words, wer = self._get_aligned_words(ref_text, streamed_text)

        matched_ref = self._match_words_to_reference(aligned_words, ground_truth)
        matched_transcripts = self._match_words_to_transcripts(
            aligned_words, output.transcripts
        )

        min_len = min(len(matched_ref), len(matched_transcripts))
        matched_ref = matched_ref[:min_len]
        matched_transcripts = matched_transcripts[:min_len]

        latencies = self._compute_latencies(
            output.chunks_processing, matched_ref, matched_transcripts
        )

        return BenchmarkResult(
            session_init_latency_ms=output.session_init_latency_ms,
            first_partial_latency_ms=latencies[0] if latencies else 0,
            per_word_latencies_ms=latencies,
            wer=wer,
        )
