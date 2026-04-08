"""
Phrase-level candidate search and overlap scoring.
"""
from __future__ import annotations
from typing import List, Tuple, Dict

from .constants import (
    CAMELOT_WHEEL, MIN_EXIT_POSITION_RATIO, MINIMUM_EXIT_ROOM_SEC
)


class OverlapScorer:
    """Score the quality of overlapping transition windows."""
    
    def score(self, ta: dict, tb: dict, a_exit_time: float,
             b_entry_time: float, trans_dur: float, spb: float) -> Dict[str, float]:
        """
        Score harmonic, vocal, rhythmic, and spectral overlap.
        
        Returns:
            Dict with keys: spectral, harmonic, vocal, rhythmic, total
        """
        scores = {}
        
        # Spectral compatibility
        scores['spectral'] = (1.0 if ta.get('energy', 'High') == tb.get('energy', 'High')
                             else 0.5)
        
        # Harmonic compatibility (Camelot)
        ca, cb = CAMELOT_WHEEL.get(ta.get('key', 'C')), CAMELOT_WHEEL.get(tb.get('key', 'C'))
        if ca and cb:
            diff = min(abs(ca[0] - cb[0]), 12 - abs(ca[0] - cb[0]))
            if diff == 0:
                scores['harmonic'] = 1.0
            elif diff == 1:
                scores['harmonic'] = 0.75
            elif diff == 2:
                scores['harmonic'] = 0.50
            else:
                scores['harmonic'] = 0.10
        else:
            scores['harmonic'] = 0.5
        
        # Vocal overlap (avoid clash)
        def overlap_fraction(regions: List[Tuple[float, float]],
                           ws: float, we: float) -> float:
            wl = we - ws
            if wl <= 0:
                return 0.0
            covered = sum(max(0.0, min(e, we) - max(s, ws))
                         for s, e in regions)
            return min(1.0, covered / wl)
        
        a_vf = overlap_fraction(
            ta.get('stems', {}).get('vocal_regions', []),
            a_exit_time, a_exit_time + trans_dur
        )
        b_vf = overlap_fraction(
            tb.get('stems', {}).get('vocal_regions', []),
            b_entry_time, b_entry_time + trans_dur
        )
        scores['vocal'] = 1.0 - (a_vf * b_vf)
        
        # Rhythmic compatibility
        bpm_ratio = (min(ta.get('bpm', 112.0), tb.get('bpm', 112.0)) /
                    max(ta.get('bpm', 112.0), tb.get('bpm', 112.0), 0.01))
        
        a_dd = (len([t for t in ta.get('stems', {}).get('log_drum_hits', [])
                    if a_exit_time <= t <= a_exit_time + trans_dur]) /
               max(trans_dur, 1))
        b_dd = (len([t for t in tb.get('stems', {}).get('log_drum_hits', [])
                    if b_entry_time <= t <= b_entry_time + trans_dur]) /
               max(trans_dur, 1))
        
        dc = min(a_dd, b_dd) / max(a_dd, b_dd, 0.01)
        scores['rhythmic'] = max(0.0, min(1.0, bpm_ratio * (1.0 - dc * 0.5)))
        
        # Weighted total
        scores['total'] = max(0.0, min(1.0,
            scores['spectral'] * 0.20 +
            scores['harmonic'] * 0.30 +
            scores['vocal'] * 0.30 +
            scores['rhythmic'] * 0.20
        ))
        
        return scores


class PhraseCandidateSearch:
    """Find optimal exit/entry phrase pairs for transitions."""
    
    def __init__(self):
        self.scorer = OverlapScorer()
    
    def search(self, ta: dict, tb: dict, b_ratio: float, trans_dur: float,
              spb: float, quiet: bool = False) -> Tuple[float, float, dict, List[dict]]:
        """
        Search for best exit/entry phrase pair.
        
        Returns:
            (best_a_exit, best_b_entry, best_score_dict, all_candidates)
        """
        # Stretch track B metadata for scoring
        tb_stretched = dict(tb)
        if abs(b_ratio - 1.0) > 0.005:
            stems_s = dict(tb.get('stems', {}))
            if 'vocal_regions' in stems_s:
                stems_s['vocal_regions'] = [
                    (s / b_ratio, e / b_ratio)
                    for s, e in stems_s['vocal_regions']
                ]
            if 'log_drum_hits' in stems_s:
                stems_s['log_drum_hits'] = [
                    t / b_ratio for t in stems_s['log_drum_hits']
                ]
            tb_stretched['stems'] = stems_s
            tb_stretched['bpm'] = tb.get('bpm', 112.0) * b_ratio
        
        a_exits = self._candidate_exits(ta, trans_dur, quiet)
        b_entries = self._candidate_entries(tb, b_ratio, trans_dur)
        
        all_candidates = []
        best_score = -1.0
        best_pair = (a_exits[0], b_entries[0])
        
        for a_exit in a_exits:
            for b_entry in b_entries:
                score = self.scorer.score(ta, tb_stretched, a_exit, b_entry,
                                         trans_dur, spb)
                all_candidates.append({
                    'a_exit': round(a_exit, 2),
                    'b_entry': round(b_entry, 2),
                    'score': score
                })
                
                if score['total'] > best_score:
                    best_score = score['total']
                    best_pair = (a_exit, b_entry)
        
        all_candidates.sort(key=lambda x: -x['score']['total'])
        best_a, best_b = best_pair
        
        if not quiet:
            print(f"   🔍 Candidate search: {len(a_exits)} exits × {len(b_entries)} entries "
                  f"= {len(all_candidates)} pairs")
            print(f"   🏆 Best pair: A={best_a:.1f}s  B={best_b:.1f}s  score={best_score:.3f}")
        
        return best_a, best_b, all_candidates[0]['score'], all_candidates
    
    def _is_safe_exit_phrase(self, p: dict, dur: float) -> bool:
        """Check if phrase is safe for exiting."""
        if dur <= 0:
            return False
        
        start = float(p.get('start', 0.0))
        pos = start / dur
        fn = p.get('function', '')
        mix = float(p.get('mixability', 0.0))
        bass = float(p.get('bass_density', 0.0))
        vocal = float(p.get('vocal_density', 0.0))
        tens = float(p.get('tension_score', 0.0))
        
        if pos < MIN_EXIT_POSITION_RATIO:
            return False
        
        # Blacklist certain phrase functions
        if fn in ('release_peak', 'bass_showcase', 'tension_build',
                 'fake_outro', 'vocal_spotlight'):
            return False
        
        # Position-dependent thresholds
        if pos >= 0.80:
            return (mix >= 0.40 and bass <= 0.75 and
                   vocal <= 0.70 and tens <= 0.80)
        
        return (mix >= 0.55 and bass <= 0.60 and
               vocal <= 0.55 and tens <= 0.65)
    
    def _score_exit_phrase(self, phrase: dict, next_phrases: List[dict],
                          track_dur: float) -> float:
        """Score phrase quality for exiting."""
        if track_dur <= 0:
            return -10.0
        
        pos = float(phrase.get('start', 0.0)) / track_dur
        mix = float(phrase.get('mixability', 0.5))
        bass = float(phrase.get('bass_density', 0.0))
        vocal = float(phrase.get('vocal_density', 0.0))
        harm = float(phrase.get('harmonic_density', 0.0))
        perc = float(phrase.get('percussive_density', 0.0))
        tens = float(phrase.get('tension_score', 0.0))
        fn = phrase.get('function', '')
        
        score = 2.5 * mix
        
        # Position bonus
        if pos > 0.85:
            score += 3.5
        elif pos > 0.80:
            score += 2.8
        elif pos > 0.75:
            score += 2.0
        elif pos > 0.70:
            score += 1.0
        else:
            score -= 1.5
        
        # Function scoring
        FN_SCORES = {
            'dj_friendly_outro': 2.5, 'decompression': 1.5,
            'drum_foundation': 1.0, 'harmonic_bed': 0.4,
            'release_peak': -3.5, 'bass_showcase': -2.5,
            'tension_build': -2.2, 'fake_outro': -2.8,
            'vocal_spotlight': -2.8
        }
        score += FN_SCORES.get(fn, 0.0)
        
        # Density penalties
        score -= 1.7 * bass + 1.5 * vocal + 1.2 * tens + 0.5 * perc
        score += 0.8 * (1.0 - harm) + 0.6 * (1.0 - perc)
        
        # Future phrase lookahead
        if next_phrases:
            fm = sum(p.get('mixability', 0.5) for p in next_phrases[:2]) / len(next_phrases[:2])
            fb = sum(p.get('bass_density', 0.0) for p in next_phrases[:2]) / len(next_phrases[:2])
            fv = sum(p.get('vocal_density', 0.0) for p in next_phrases[:2]) / len(next_phrases[:2])
            ft = sum(p.get('tension_score', 0.0) for p in next_phrases[:2]) / len(next_phrases[:2])
            
            score += 0.8 * fm - 1.0 * fb - 0.8 * fv - 1.0 * ft
            score -= self._future_rebound_penalty(next_phrases)
        
        return score
    
    def _future_rebound_penalty(self, next_phrases: List[dict]) -> float:
        """Penalize upcoming intensity spikes."""
        penalty = 0.0
        for p in next_phrases[:2]:
            fn = p.get('function', '')
            if fn == 'release_peak':
                penalty += 2.5
            elif fn == 'bass_showcase':
                penalty += 2.0
            elif fn == 'tension_build':
                penalty += 1.5
            elif fn == 'vocal_spotlight':
                penalty += 1.5
            elif fn == 'fake_outro':
                penalty += 2.0
            
            penalty += (1.2 * float(p.get('bass_density', 0.0)) +
                       0.8 * float(p.get('vocal_density', 0.0)) +
                       1.0 * float(p.get('tension_score', 0.0)) +
                       0.6 * float(p.get('percussive_density', 0.0)))
        
        return penalty
    
    def _candidate_exits(self, ta: dict, trans_dur: float,
                        quiet: bool = False) -> List[float]:
        """Generate candidate exit points for track A."""
        dur = ta.get('_duration', 300.0)
        exit_room = min(trans_dur, MINIMUM_EXIT_ROOM_SEC)
        
        phrases = (ta.get('phrases', []) or ta.get('phrase_map', []) or
                  ta.get('phrase_analysis', []) or ta.get('phrase_windows', []))
        
        if phrases:
            # Score all valid phrases
            scored = []
            for i, p in enumerate(phrases):
                start = float(p.get('start', 0.0))
                
                if (start < dur * MIN_EXIT_POSITION_RATIO or
                    start + exit_room >= dur or
                    not self._is_safe_exit_phrase(p, dur)):
                    continue
                
                s = self._score_exit_phrase(p, phrases[i+1:i+3], dur)
                scored.append((s, start, p))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            if scored:
                result = [round(start, 2) for _s, start, _p in scored[:5]]
                safety_exit = round(max(30.0, dur - exit_room - 2.0), 2)
                if not any(abs(r - safety_exit) < 5.0 for r in result):
                    result.append(safety_exit)
                return sorted(result)
            
            # Fallback: latest safe phrase
            late = [(float(p.get('start', 0.0)), p) for p in phrases
                   if (float(p.get('start', 0.0)) / max(dur, 1.0) >= MIN_EXIT_POSITION_RATIO
                       and float(p.get('start', 0.0)) + exit_room < dur)]
            if late:
                result = [round(max(late, key=lambda x: x[0])[0], 2)]
                safety_exit = round(max(30.0, dur - exit_room - 2.0), 2)
                if not any(abs(r - safety_exit) < 5.0 for r in result):
                    result.append(safety_exit)
                return sorted(result)
        
        # Fallback: metadata-based exits
        exits_set = set()
        for pt in ta.get('best_exit_phrases', []):
            exits_set.add(round(float(pt), 1))
        
        prep = ta.get('zones', {}).get('optimal_mix_out')
        if prep and prep > dur * MIN_EXIT_POSITION_RATIO:
            exits_set.add(round(float(prep), 1))
        
        for sec in ta.get('structure_map', []):
            if (sec.get('label') in ('breakdown', 'outro') and
                sec.get('start', 0) > dur * MIN_EXIT_POSITION_RATIO):
                exits_set.add(round(float(sec['start']), 1))
        
        valid = sorted(e for e in exits_set
                      if e > (dur * MIN_EXIT_POSITION_RATIO)
                      and e + exit_room < dur)
        
        safety_exit = round(max(30.0, dur - exit_room - 2.0), 2)
        result = valid[:5]
        if not any(abs(r - safety_exit) < 5.0 for r in result):
            result.append(safety_exit)
        
        return sorted(result)
    
    def _candidate_entries(self, tb: dict, b_ratio: float,
                          trans_dur: float) -> List[float]:
        """Generate candidate entry points for track B."""
        entries = set([round(max(0.0, tb.get('first_beat_time', 0.0) / b_ratio), 1)])
        
        for pt in tb.get('best_entry_phrases', []):
            entries.add(round(pt / b_ratio, 1))
        
        for sec in tb.get('structure_map', []):
            if sec.get('label') == 'intro':
                entries.add(round(sec.get('end', 0) / b_ratio, 1))
            if (sec.get('label') == 'breakdown' and
                sec.get('start', 0) / b_ratio < trans_dur * 0.4):
                entries.add(round(sec['start'] / b_ratio, 1))
        
        if tb.get('piano_entries', []):
            entries.add(round(tb['piano_entries'][0] / b_ratio, 1))
        
        b_dur = tb.get('_duration', 300.0) / b_ratio
        valid = sorted(e for e in entries
                      if e >= 0 and e + trans_dur + 60.0 < b_dur)
        
        return (valid[:5] if valid
               else [round(max(0.0, tb.get('first_beat_time', 0.0) / b_ratio), 1)])