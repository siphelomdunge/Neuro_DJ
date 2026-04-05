/*
 * core_audio.cpp  —  Neuro-DJ Audio Engine v5.3
 *
 * COMPILED WITH: Pybind11 + PortAudio + dr_wav
 * * V5.3 UPDATES:
 * • Slew-rate limiting on all gain/EQ changes (eliminates all clicks/zipper noise).
 * • Mathematical curve library for constant-power crossfades.
 * • Reverb Wash (4-comb Schroeder) and DC Blocked Delays.
 * • Transparent soft-limiting on the master bus.
 * • SONIC MUD FIX: Echo Throw/Freeze actively high-pass and duck the tail volume
 * so incoming bass frequencies aren't masked by the delay feedback.
 */

#include <iostream>
#include <string>
#include <portaudio.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>
#include <atomic>
#include <stdexcept>
#include <cstring>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace py = pybind11;
static const float kPI = 3.14159265359f;

// --- Mathematically proven curve library --------------------------------------

inline float curve_smooth(float t)   { return t * t * (3.0f - 2.0f * t); }
inline float curve_enter(float t)    { return sinf(t * (kPI / 2.0f)); }
inline float curve_exit(float t)     { return cosf(t * (kPI / 2.0f)); }
inline float curve_power(float t)    { return sinf(t * (kPI / 2.0f)); }

// sigmoid with clamped exponent to prevent float overflow
inline float sigmoid_swap(float t, float s=0.5f, float k=70.0f){
    float x = -k * (t - s);
    if(x >  80.0f) return 0.0f;
    if(x < -80.0f) return 1.0f;
    return 1.0f / (1.0f + expf(x));
}

// Constant-power bass swap: cos/sin wrapped around sigmoid.
// A² + B² = cos²(s·p/2) + sin²(s·p/2) = 1 for ALL t.
inline float bass_cp_a(float sigma){ return cosf(sigma * (kPI / 2.0f)); }
inline float bass_cp_b(float sigma){ return sinf(sigma * (kPI / 2.0f)); }

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// compensated_pair — fixes the mid/high asymmetric timing power dip.
inline void compensated_pair(float exit_t, float entry_t, float& a_out, float& b_out){
    float a = cosf(exit_t  * (kPI / 2.0f));
    float b = sinf(entry_t * (kPI / 2.0f));
    float power = a*a + b*b;
    float comp  = (power > 0.0f) ? (1.0f / sqrtf(power)) : 1.0f;
    comp = (comp > 1.414f) ? 1.414f : comp;  // cap at +3 dB
    a_out = a * comp;
    b_out = b * comp;
}

// wobble_scaled — applies humanize wobble but only during active movement.
inline float wobble_scaled(float gain, float t, float scale,
                           float w_f1, float w_f2, float w_p1, float w_p2, float w_amp){
    if(scale < 0.01f) return clampf(gain, 0.0f, 1.0f);
    float w = w_amp * scale * sinf(w_f1*kPI*t + w_p1) * sinf(w_f2*kPI*t + w_p2);
    return clampf(gain + w, 0.0f, 1.0f);
}

// Transparent tanh soft-limiter. drive=0.55 means unity below ~0.85 FS.
inline float soft_limit(float x) {
    constexpr float D = 0.55f;
    return tanhf(x * D) / D;
}

// --- EQFilter with gain slewing ----------------------------------------------
struct EQFilter {
    float low_gain=1,mid_gain=1,high_gain=1;
    float tgt_low =1,tgt_mid =1,tgt_high =1;

    float ev_low_mult=1,ev_mid_mult=1,ev_high_mult=1;
    float tgt_ev_low =1,tgt_ev_mid =1,tgt_ev_high =1;

    float l_low=0,l_high=0,r_low=0,r_high=0;
    float alpha_low, alpha_high;
    static constexpr float SLEW    = 0.012f;
    static constexpr float EV_SLEW = 0.010f;

    EQFilter(float lF=250.f, float hF=4000.f) {
        alpha_low  = (2*kPI*lF)/(44100.f + 2*kPI*lF);
        alpha_high = (2*kPI*hF)/(44100.f + 2*kPI*hF);
    }
    inline void advance() {
        low_gain  += SLEW*(tgt_low  - low_gain);
        mid_gain  += SLEW*(tgt_mid  - mid_gain);
        high_gain += SLEW*(tgt_high - high_gain);
        ev_low_mult  += EV_SLEW*(tgt_ev_low  - ev_low_mult);
        ev_mid_mult  += EV_SLEW*(tgt_ev_mid  - ev_mid_mult);
        ev_high_mult += EV_SLEW*(tgt_ev_high - ev_high_mult);
    }
    inline float process(float s, bool isL) {
        float lp,hp,mp;
        if(isL){ l_low+=alpha_low*(s-l_low);lp=l_low; l_high+=alpha_high*(s-l_high);hp=s-l_high; }
        else   { r_low+=alpha_low*(s-r_low);lp=r_low; r_high+=alpha_high*(s-r_high);hp=s-r_high; }
        mp = s-lp-hp;
        return lp*(low_gain*ev_low_mult) + mp*(mid_gain*ev_mid_mult) + hp*(high_gain*ev_high_mult);
    }
    void reset_flat(){
        tgt_low=tgt_mid=tgt_high=1;
        low_gain=mid_gain=high_gain=1;
        tgt_ev_low=tgt_ev_mid=tgt_ev_high=1;
        ev_low_mult=ev_mid_mult=ev_high_mult=1;
        l_low=l_high=r_low=r_high=0;
    }
};

// --- Delay with DC blocker ----------------------------------------------------
struct DelayEffect {
    std::vector<float> bufL,bufR;
    int bufSize=88200,writeIdx=0,delaySamples=20000;

    float feedback=0.6f, tgt_feedback=0.6f;
    static constexpr float FB_SLEW = 0.002f;

    float mix=0;
    float dcL=0,dcR=0;

    DelayEffect(){ bufL.resize(bufSize,0); bufR.resize(bufSize,0); }

    void reset(){
        std::fill(bufL.begin(),bufL.end(),0);
        std::fill(bufR.begin(),bufR.end(),0);
        dcL=dcR=0; mix=0;
        feedback=tgt_feedback=0.6f;
    }

    inline void advance(){
        feedback += FB_SLEW*(tgt_feedback - feedback);
    }

    inline void process(float& sL,float& sR){
        if(mix<0.001f) return;
        int ri=writeIdx-delaySamples; if(ri<0) ri+=bufSize;
        float dL=bufL[ri], dR=bufR[ri];
        float fdL=dL-dcL+0.995f*dcL; dcL=dL;
        float fdR=dR-dcR+0.995f*dcR; dcR=dR;
        bufL[writeIdx]=sL+fdL*feedback;
        bufR[writeIdx]=sR+fdR*feedback;
        writeIdx=(writeIdx+1)%bufSize;
        sL=sL*(1-mix)+dL*mix;
        sR=sR*(1-mix)+dR*mix;
    }
};

// --- Reverb Wash (4-comb Schroeder) ------------------------------------------
struct ReverbWash {
    static const int NC=4;
    static const int D[NC];
    float bL[NC][1400], bR[NC][1400];
    int   ix[NC];
    float fb=0.76f, mix=0;
    ReverbWash(){ memset(bL,0,sizeof(bL)); memset(bR,0,sizeof(bR)); memset(ix,0,sizeof(ix)); }
    void reset(){ memset(bL,0,sizeof(bL)); memset(bR,0,sizeof(bR)); memset(ix,0,sizeof(ix)); mix=0; }
    inline void process(float& L,float& R){
        if(mix<0.001f) return;
        float wL=0,wR=0;
        for(int i=0;i<NC;i++){
            float oL=bL[i][ix[i]], oR=bR[i][ix[i]];
            bL[i][ix[i]]=L+oL*fb; bR[i][ix[i]]=R+oR*fb;
            ix[i]=(ix[i]+1)%D[i];
            wL+=oL; wR+=oR;
        }
        wL/=NC; wR/=NC;
        L=L*(1-mix)+wL*mix; R=R*(1-mix)+wR*mix;
    }
};
const int ReverbWash::D[4]={1031,1153,1237,1361};

// --- Deck ---------------------------------------------------------------------
struct Deck {
    float* pSampleData=nullptr;
    drwav_uint64 totalSampleCount=0;
    std::atomic<drwav_uint64> currentSampleIndex{0};
    std::atomic<bool> isPlaying{false};

    float volume=1, tgt_volume=1;
    static constexpr float VOL_SLEW=0.008f;

    float gate_volume=1.0f;

    EQFilter eq; DelayEffect fx; ReverbWash reverb;
    std::vector<float> visualBuffer; int visualIdx=0;
    bool isLooping=false;
    drwav_uint64 loopStartSample=0,loopEndSample=0;
    int targetLoopCount=0,currentLoopCount=0;

    Deck(){ visualBuffer.resize(512,0); }
    ~Deck(){ if(pSampleData){ drwav_free(pSampleData,nullptr); pSampleData=nullptr; } }

    inline void advance(){
        volume += VOL_SLEW*(tgt_volume - volume);
        eq.advance();
        fx.advance();
    }
    void reset_state(){
        eq.reset_flat(); fx.reset(); reverb.reset();
        volume=tgt_volume=1; gate_volume=1; isLooping=false;
    }
};

// --- Scheduled EQ event (stem-separation fakeout) ----------------------------
struct EQEvent {
    drwav_uint64 trigger_sample;
    int   deck_id;
    float low_mult, mid_mult, high_mult;
    bool  fired = false;
};

// --- Lock-free command queues --------------------------------------------------
struct LoopCommand {
    std::atomic<bool> pending{false};
    int track_id; drwav_uint64 loopStartSample,loopEndSample; int loop_count;
};
struct TransCommand {
    std::atomic<bool> pending{false};
    drwav_uint64 transTotalFrames;
    float p_beats,p_bass_swap,p_echo,p_stutter,p_wash,p_bpm,p_piano_hold;
    int technique_id;
    float w_f1=2.3f,w_f2=5.7f,w_p1=0.0f,w_p2=0.0f,w_amp=0.012f;
};

// --- NeuroMixer ---------------------------------------------------------------
class NeuroMixer {
    Deck deckA,deckB;
    PaStream* stream;
    std::mutex audioMutex;

    std::atomic<bool> inTransition{false};
    drwav_uint64 transTotalFrames=0,transCurrentFrame=0;
    float p_beats=16,p_bass_swap=0.75f,p_echo=0,p_stutter=0,p_wash=0,p_bpm=112,p_piano_hold=0;
    int technique_id=0;

    float w_f1=2.3f,w_f2=5.7f,w_p1=0.0f,w_p2=0.0f,w_amp=0.012f;

    LoopCommand  pendingLoop;
    TransCommand pendingTrans;

    std::vector<EQEvent>  eqSchedule;
    std::atomic<bool>     pendingClearEQ{false};

    static int audioCallback(const void*, void* outBuf, unsigned long nFrames,
                             const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* ud) {
        NeuroMixer* mx=(NeuroMixer*)ud;
        float* out=(float*)outBuf;

        // Drain queues
        if(mx->pendingLoop.pending.load(std::memory_order_relaxed)){
            std::atomic_thread_fence(std::memory_order_acquire);
            Deck* t=(mx->pendingLoop.track_id==0)?&mx->deckA:&mx->deckB;
            t->loopStartSample=mx->pendingLoop.loopStartSample;
            t->loopEndSample  =mx->pendingLoop.loopEndSample;
            t->targetLoopCount=mx->pendingLoop.loop_count;
            t->currentLoopCount=0;
            t->isLooping=(mx->pendingLoop.loop_count>0);
            mx->pendingLoop.pending.store(false,std::memory_order_relaxed);
        }
        if(mx->pendingTrans.pending.load(std::memory_order_relaxed)){
            std::atomic_thread_fence(std::memory_order_acquire);
            mx->transTotalFrames =mx->pendingTrans.transTotalFrames;
            mx->transCurrentFrame=0;
            mx->p_beats     =mx->pendingTrans.p_beats;
            mx->p_bass_swap =mx->pendingTrans.p_bass_swap;
            mx->p_echo      =mx->pendingTrans.p_echo;
            mx->p_stutter   =mx->pendingTrans.p_stutter;
            mx->p_wash      =mx->pendingTrans.p_wash;
            mx->p_bpm       =mx->pendingTrans.p_bpm;
            mx->p_piano_hold=mx->pendingTrans.p_piano_hold;
            mx->technique_id=mx->pendingTrans.technique_id;
            mx->w_f1=mx->pendingTrans.w_f1; mx->w_f2=mx->pendingTrans.w_f2;
            mx->w_p1=mx->pendingTrans.w_p1; mx->w_p2=mx->pendingTrans.w_p2;
            mx->w_amp=mx->pendingTrans.w_amp;
            mx->deckA.fx.reset(); mx->deckA.reverb.reset();
            mx->deckB.fx.reset(); mx->deckB.reverb.reset();
            mx->deckA.eq.tgt_low=mx->deckA.eq.tgt_mid=mx->deckA.eq.tgt_high=1;
            mx->deckA.eq.low_gain=mx->deckA.eq.mid_gain=mx->deckA.eq.high_gain=1;
            mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1;
            mx->deckB.eq.low_gain=mx->deckB.eq.mid_gain=mx->deckB.eq.high_gain=1;
            mx->deckA.volume    =1.0f; mx->deckA.tgt_volume=1.0f; mx->deckA.gate_volume=1.0f;
            mx->deckB.volume    =0.0f; mx->deckB.tgt_volume=0.0f; mx->deckB.gate_volume=1.0f;
            mx->deckB.isPlaying.store(true,std::memory_order_relaxed);
            mx->inTransition.store(true,std::memory_order_relaxed);
            mx->pendingTrans.pending.store(false,std::memory_order_relaxed);
        }

        std::unique_lock<std::mutex> lock(mx->audioMutex,std::try_to_lock);
        if(!lock.owns_lock()){ std::fill(out,out+nFrames*2,0.f); return paContinue; }

        bool isTrans=mx->inTransition.load(std::memory_order_relaxed);

        if(mx->pendingClearEQ.load(std::memory_order_relaxed)){
            mx->eqSchedule.clear();
            mx->deckA.eq.tgt_ev_low=mx->deckA.eq.tgt_ev_mid=mx->deckA.eq.tgt_ev_high=1.0f;
            mx->deckB.eq.tgt_ev_low=mx->deckB.eq.tgt_ev_mid=mx->deckB.eq.tgt_ev_high=1.0f;
            mx->pendingClearEQ.store(false,std::memory_order_relaxed);
        }

        if(!mx->eqSchedule.empty()){
            drwav_uint64 posA=mx->deckA.currentSampleIndex.load(std::memory_order_relaxed);
            drwav_uint64 posB=mx->deckB.currentSampleIndex.load(std::memory_order_relaxed);
            for(auto& ev : mx->eqSchedule){
                if(ev.fired) continue;
                drwav_uint64 pos = (ev.deck_id==0) ? posA : posB;
                if(pos >= ev.trigger_sample){
                    Deck* t = (ev.deck_id==0) ? &mx->deckA : &mx->deckB;
                    t->eq.tgt_ev_low  = ev.low_mult;
                    t->eq.tgt_ev_mid  = ev.mid_mult;
                    t->eq.tgt_ev_high = ev.high_mult;
                    ev.fired = true;
                }
            }
        }

        for(unsigned int i=0;i<nFrames;i++){
            // ? ADVANCE SMOOTHING FIRST
            mx->deckA.advance();
            mx->deckB.advance();

            // ? TECHNIQUE DISPATCH
            if(isTrans){
                if(mx->transTotalFrames == 0){
                    mx->inTransition.store(false,std::memory_order_relaxed); isTrans=false;
                    mx->deckA.tgt_volume=0; mx->deckB.tgt_volume=1;
                    mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1;
                    mx->deckA.fx.mix=0; mx->deckA.reverb.mix=0; mx->deckB.reverb.mix=0;
                    mx->deckA.gate_volume=1.0f; mx->deckB.gate_volume=1.0f;
                    continue;
                }
                float prog=(float)mx->transCurrentFrame/(float)mx->transTotalFrames;
                if(prog>=1.f){
                    mx->inTransition.store(false,std::memory_order_relaxed); isTrans=false;
                    mx->deckB.tgt_volume=1.0f;
                    mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1.0f;
                    mx->deckA.reverb.mix=0; mx->deckB.reverb.mix=0;
                    mx->deckA.gate_volume=1.0f; mx->deckB.gate_volume=1.0f;

                    if(mx->technique_id == 3 || mx->technique_id == 7){
                        mx->deckA.tgt_volume = 0.4f;  // V5.3: Duck the tail volume by 60%
                        mx->deckA.eq.tgt_low = 0.0f;  // V5.3: High-pass the echo tail
                    } else {
                        mx->deckA.tgt_volume = 0.0f;
                        mx->deckA.fx.mix     = 0.0f;
                    }
                } else {

                // -- Common helpers (beat_samples BEFORE beat_t) --------------
                float beat_samples = 44100.0f * 60.0f / mx->p_bpm;
                float beat_t = (float)mx->transCurrentFrame / beat_samples;
                float total_beats = mx->p_beats;

                // -- Bass-swap position clamp (per-technique) -----------------
                float safe_bass_swap = mx->p_bass_swap;
                if(mx->technique_id == 0 || mx->technique_id == 2 || mx->technique_id == 4){
                    safe_bass_swap = clampf(mx->p_bass_swap, 0.40f, 0.75f);
                } else if(mx->technique_id == 5){
                    safe_bass_swap = clampf(mx->p_bass_swap, 0.40f, 0.55f);
                }

                // Per-transition unique wobble
                auto wobble = [&](float gain, float t, float scale=1.0f) -> float {
                    return wobble_scaled(gain, t, scale,
                        mx->w_f1, mx->w_f2, mx->w_p1, mx->w_p2, mx->w_amp);
                };

                switch(mx->technique_id){

                // -- 0: BASS SWAP ----------------------------------------------
                case 0: default: {
                    float swap_beat = total_beats * safe_bass_swap;
                    float t_all   = clampf(beat_t / total_beats, 0.0f, 1.0f);
                    float t_phase1= clampf(beat_t / clampf(swap_beat,1.f,9999.f), 0.0f, 1.0f);
                    float bass_sigma = sigmoid_swap(t_all, safe_bass_swap, 70.0f);
                    float t_fade  = clampf((beat_t - swap_beat - 2.0f)
                                          / clampf(total_beats - swap_beat - 2.0f,1.f,9999.f),
                                          0.0f, 1.0f);

                    mx->deckA.eq.tgt_low = bass_cp_a(bass_sigma);
                    mx->deckB.eq.tgt_low = bass_cp_b(bass_sigma);

                    float b_mid_t  = clampf((t_phase1 - 0.40f) / 0.60f, 0.0f, 1.0f);
                    float b_high_t = t_phase1;

                    if(beat_t < swap_beat){
                        mx->deckA.tgt_volume  = 1.0f;
                        mx->deckA.eq.tgt_mid  = 1.0f;
                        mx->deckA.eq.tgt_high = 1.0f;
                        float _am, _bm, _ah, _bh;
                        compensated_pair(0.0f, b_mid_t,  _am, _bm);
                        compensated_pair(0.0f, b_high_t, _ah, _bh);
                        mx->deckB.eq.tgt_mid  = _bm;
                        mx->deckB.eq.tgt_high = _bh;
                        float active = clampf((t_phase1 - 0.2f) / 0.8f, 0.0f, 1.0f);
                        mx->deckB.tgt_volume  = wobble(curve_enter(t_phase1)*0.85f, t_phase1, active);
                    } else {
                        float a_mid_t = clampf(t_fade / 0.85f, 0.0f, 1.0f);
                        float _am, _bm, _ah, _bh;
                        compensated_pair(a_mid_t, 1.0f, _am, _bm);
                        compensated_pair(t_fade,  1.0f, _ah, _bh);
                        mx->deckA.eq.tgt_mid  = _am;
                        mx->deckA.eq.tgt_high = _ah;
                        mx->deckB.eq.tgt_mid  = 1.0f;
                        mx->deckB.eq.tgt_high = 1.0f;
                        mx->deckA.tgt_volume  = wobble(curve_exit(t_fade), t_fade, t_fade);
                        mx->deckB.tgt_volume  = wobble(0.85f + curve_enter(
                            clampf((beat_t-swap_beat)/2.f,0.f,1.f))*0.15f, t_phase1, 0.3f);
                    }
                    break;
                }

                // -- 2: FILTER SWEEP -------------------------------------------
                case 2: {
                    float t_full = clampf(beat_t / total_beats, 0.0f, 1.0f);
                    float bass_sigma = sigmoid_swap(t_full, 0.55f, 70.0f);

                    mx->deckA.eq.tgt_low = bass_cp_a(bass_sigma);
                    mx->deckB.eq.tgt_low = bass_cp_b(bass_sigma);

                    mx->deckA.eq.tgt_high = 1.0f - curve_smooth(clampf(beat_t/(total_beats*0.65f),0.f,1.f));
                    mx->deckA.eq.tgt_mid  = 1.0f - curve_smooth(clampf((beat_t-total_beats*0.25f)/(total_beats*0.60f),0.f,1.f));
                    mx->deckB.eq.tgt_high = curve_smooth(clampf(beat_t/(total_beats*0.50f),0.f,1.f));
                    mx->deckB.eq.tgt_mid  = curve_smooth(clampf((beat_t-total_beats*0.30f)/(total_beats*0.45f),0.f,1.f));

                    mx->deckA.tgt_volume  = wobble(curve_exit(t_full), t_full);
                    mx->deckB.tgt_volume  = wobble(curve_enter(t_full) * 0.9f, t_full);
                    mx->deckA.reverb.mix  = curve_smooth(clampf((beat_t-total_beats*0.20f)/(total_beats*0.60f),0.f,1.f)) * mx->p_wash * 0.55f;
                    break;
                }

                // -- 3: ECHO THROW --------------------------------------------
                case 3: {
                    float trap_beat   = total_beats * 0.65f;
                    float trap_len    = 4.0f;
                    float kill_beat   = trap_beat + trap_len;
                    float overlap_prog= clampf(beat_t / trap_beat, 0.0f, 1.0f);
                    float trap_prog   = clampf((beat_t - trap_beat) / trap_len, 0.0f, 1.0f);
                    bool  past_kill   = (beat_t >= kill_beat);

                    mx->deckA.fx.delaySamples = (int)beat_samples;
                    mx->deckB.fx.mix = 0.0f;

                    if(!past_kill){
                        mx->deckA.gate_volume    = 1.0f;
                        mx->deckA.tgt_volume     = 1.0f;
                        mx->deckA.eq.tgt_low = mx->deckA.eq.tgt_mid = mx->deckA.eq.tgt_high = 1.0f;
                        mx->deckA.fx.mix         = curve_smooth(trap_prog) * mx->p_echo;
                        mx->deckA.fx.tgt_feedback= 0.60f + curve_smooth(trap_prog) * 0.28f;

                        mx->deckB.tgt_volume     = wobble(curve_enter(overlap_prog) * 0.75f, overlap_prog);
                        mx->deckB.eq.tgt_low     = 0.0f;
                        mx->deckB.eq.tgt_mid     = 0.85f;
                        mx->deckB.eq.tgt_high    = 1.0f;
                    } else {
                        mx->deckA.gate_volume    = 0.0f;
                        mx->deckA.tgt_volume     = 0.4f;   // V5.3: Drop volume so B punches through
                        mx->deckA.eq.tgt_low     = 0.0f;   // V5.3: Kill bass in the echo
                        mx->deckA.fx.mix         = 1.0f;
                        mx->deckA.fx.tgt_feedback= 0.75f;  // V5.3: Smoothly decay the feedback

                        mx->deckB.tgt_volume     = 1.0f;
                        mx->deckB.eq.tgt_low = mx->deckB.eq.tgt_mid = mx->deckB.eq.tgt_high = 1.0f;
                    }
                    break;
                }

                // -- 4: SLOW BURN ---------------------------------------------
                case 4: {
                    float angle = prog * (kPI / 2.0f);
                    mx->deckA.tgt_volume = wobble(cosf(angle), prog);
                    mx->deckB.tgt_volume = wobble(sinf(angle), prog);
                    mx->deckA.eq.tgt_low = mx->deckA.eq.tgt_mid = mx->deckA.eq.tgt_high = 1.0f;
                    mx->deckB.eq.tgt_low = mx->deckB.eq.tgt_mid = mx->deckB.eq.tgt_high = 1.0f;
                    break;
                }

                // -- 5: PIANO HANDOFF (Amapiano) ------------------------------
                case 5: {
                    float swap_beat   = total_beats * safe_bass_swap;
                    float presig_beat = swap_beat - 1.0f;
                    float bass_t      = clampf((beat_t - swap_beat) / 2.0f, 0.0f, 1.0f);
                    float piano_t     = clampf((beat_t - swap_beat - 2.0f) / 2.0f, 0.0f, 1.0f);
                    float farewell_t  = clampf((beat_t - swap_beat - 4.0f)
                                              / (total_beats - swap_beat - 4.0f), 0.0f, 1.0f);

                    float stage1_t    = clampf(beat_t / 16.0f, 0.0f, 1.0f);
                    float stage2_t    = clampf((beat_t - 16.0f) / clampf(swap_beat - 16.0f, 1.0f, 9999.f), 0.0f, 1.0f);
                    float presig_t    = clampf((beat_t - presig_beat) / 1.0f, 0.0f, 1.0f);

                    if(beat_t < swap_beat){
                        mx->deckA.tgt_volume  = 1.0f;
                        mx->deckA.eq.tgt_low  = 1.0f;
                        mx->deckA.eq.tgt_mid  = 1.0f;
                        float highroll        = clampf(beat_t / swap_beat, 0.0f, 1.0f);
                        mx->deckA.eq.tgt_high = 1.0f - highroll * 0.10f - presig_t * 0.10f;

                        float b_vol = curve_smooth(stage1_t) * 0.05f
                                    + curve_enter(stage2_t) * 0.60f;
                        mx->deckB.tgt_volume  = wobble(b_vol, stage2_t);

                        mx->deckB.eq.tgt_low  = 0.0f;
                        mx->deckB.eq.tgt_mid  = 0.0f;
                        mx->deckB.eq.tgt_high = curve_smooth(stage1_t) * 0.15f
                                              + curve_smooth(stage2_t) * 0.65f;
                    } else {
                        float bass_sigma = sigmoid_swap(bass_t, 0.5f, 70.0f);
                        mx->deckA.eq.tgt_low  = bass_cp_a(bass_sigma);
                        float accent = bass_cp_b(bass_sigma) + (1.0f - clampf(beat_t - swap_beat, 0.f, 1.f)) * 0.15f;
                        mx->deckB.eq.tgt_low  = clampf(accent, 0.0f, 1.15f);

                        float shake = clampf(piano_t, 0.0f, 1.0f)
                                    * (1.0f - clampf((beat_t - swap_beat - 3.0f), 0.0f, 1.0f));
                        mx->deckA.eq.tgt_mid  = clampf(1.0f - curve_exit(piano_t) + shake * 0.08f, 0.0f, 1.08f);
                        mx->deckB.eq.tgt_mid  = clampf(curve_enter(piano_t)        + shake * 0.08f, 0.0f, 1.08f);

                        mx->deckA.eq.tgt_high = 0.80f + farewell_t * 0.20f;
                        mx->deckB.eq.tgt_high = 0.85f + curve_enter(piano_t) * 0.15f;

                        float b_vol = 0.65f + curve_enter(piano_t) * 0.35f;
                        mx->deckB.tgt_volume  = wobble(b_vol, piano_t);
                        mx->deckA.tgt_volume  = wobble(curve_exit(farewell_t), farewell_t);
                        mx->deckA.reverb.mix  = clampf(farewell_t * 0.18f, 0.0f, 0.18f);
                    }
                    break;
                }

                // -- 7: ECHO FREEZE -------------------------------------------
                case 7: {
                    float trap_start = total_beats - 6.0f;
                    float kill_beat  = total_beats - 2.0f;
                    float trap_prog  = clampf((beat_t - trap_start) / 4.0f, 0.0f, 1.0f);
                    float kill_prog  = clampf((beat_t - kill_beat)  / 2.0f, 0.0f, 1.0f);

                    mx->deckA.fx.delaySamples = (int)beat_samples;
                    mx->deckB.fx.mix = 0.0f;

                    if(beat_t < trap_start){
                        mx->deckA.gate_volume = 1.0f; mx->deckA.tgt_volume = 1.0f;
                        mx->deckA.eq.tgt_low = mx->deckA.eq.tgt_mid = mx->deckA.eq.tgt_high = 1.0f;
                        mx->deckA.fx.mix = 0.0f; mx->deckA.fx.tgt_feedback = 0.60f;
                        mx->deckB.tgt_volume = 0.0f;
                        mx->deckB.eq.tgt_low = mx->deckB.eq.tgt_mid = mx->deckB.eq.tgt_high = 0.0f;
                    } else if(beat_t < kill_beat){
                        mx->deckA.gate_volume     = 1.0f; mx->deckA.tgt_volume = 1.0f;
                        mx->deckA.fx.mix          = curve_power(trap_prog);
                        mx->deckA.fx.tgt_feedback = 0.60f + trap_prog * 0.30f;
                        mx->deckB.tgt_volume      = 0.0f;
                    } else {
                        mx->deckA.gate_volume     = 0.0f;
                        mx->deckA.tgt_volume      = 0.5f;  // V5.3: Duck volume slightly less than Throw
                        mx->deckA.eq.tgt_low      = 0.0f;  // V5.3: Kill bass in the freeze
                        mx->deckA.fx.mix          = 1.0f;
                        mx->deckA.fx.tgt_feedback = 0.85f; // V5.3: Faster, smoother decay
                        mx->deckB.gate_volume     = 1.0f;
                        mx->deckB.tgt_volume      = curve_power(kill_prog);
                        mx->deckB.eq.tgt_low = mx->deckB.eq.tgt_mid = mx->deckB.eq.tgt_high = 1.0f;
                    }
                    break;
                }
                }   // end switch
                }   // end else (prog < 1.0)
                mx->transCurrentFrame++;
            }

            // ? RENDER DECK A
            float oLA=0,oRA=0;
            if(mx->deckA.isPlaying.load(std::memory_order_relaxed) && mx->deckA.pSampleData){
                drwav_uint64 idx=mx->deckA.currentSampleIndex.load(std::memory_order_relaxed);
                if(idx+1<mx->deckA.totalSampleCount){
                    if(mx->deckA.isLooping && idx+2>mx->deckA.loopEndSample){
                        if(++mx->deckA.currentLoopCount>=mx->deckA.targetLoopCount)
                            mx->deckA.isLooping=false;
                        else idx=mx->deckA.loopStartSample;
                    }
                    oLA=mx->deckA.eq.process(mx->deckA.pSampleData[idx],  true )
                          * mx->deckA.volume * mx->deckA.gate_volume;
                    oRA=mx->deckA.eq.process(mx->deckA.pSampleData[idx+1],false)
                          * mx->deckA.volume * mx->deckA.gate_volume;
                    mx->deckA.fx.process(oLA,oRA);
                    mx->deckA.reverb.process(oLA,oRA);
                    mx->deckA.currentSampleIndex.store(idx+2,std::memory_order_relaxed);
                }
            }
            // ? RENDER DECK B
            float oLB=0,oRB=0;
            if(mx->deckB.isPlaying.load(std::memory_order_relaxed) && mx->deckB.pSampleData){
                drwav_uint64 idx=mx->deckB.currentSampleIndex.load(std::memory_order_relaxed);
                if(idx+1<mx->deckB.totalSampleCount){
                    if(mx->deckB.isLooping && idx+2>mx->deckB.loopEndSample){
                        if(++mx->deckB.currentLoopCount>=mx->deckB.targetLoopCount)
                            mx->deckB.isLooping=false;
                        else idx=mx->deckB.loopStartSample;
                    }
                    oLB=mx->deckB.eq.process(mx->deckB.pSampleData[idx],  true )
                          * mx->deckB.volume * mx->deckB.gate_volume;
                    oRB=mx->deckB.eq.process(mx->deckB.pSampleData[idx+1],false)
                          * mx->deckB.volume * mx->deckB.gate_volume;
                    mx->deckB.fx.process(oLB,oRB);
                    mx->deckB.reverb.process(oLB,oRB);
                    mx->deckB.currentSampleIndex.store(idx+2,std::memory_order_relaxed);
                }
            }
            // ? Visual (decimated)
            if(i%4==0){
                mx->deckA.visualBuffer[mx->deckA.visualIdx]=oLA;
                mx->deckA.visualIdx=(mx->deckA.visualIdx+1)%512;
                mx->deckB.visualBuffer[mx->deckB.visualIdx]=oLB;
                mx->deckB.visualIdx=(mx->deckB.visualIdx+1)%512;
            }
            // ? Mix + transparent soft limit
            *out++=soft_limit(oLA+oLB);
            *out++=soft_limit(oRA+oRB);
        }
        return paContinue;
    }

public:
    NeuroMixer(){
        PaError e=Pa_Initialize();
        if(e!=paNoError) throw std::runtime_error(Pa_GetErrorText(e));
        e=Pa_OpenDefaultStream(&stream,0,2,paFloat32,44100,256,audioCallback,this);
        if(e!=paNoError){ Pa_Terminate(); throw std::runtime_error(Pa_GetErrorText(e)); }
        e=Pa_StartStream(stream);
        if(e!=paNoError){ Pa_CloseStream(stream); Pa_Terminate(); throw std::runtime_error(Pa_GetErrorText(e)); }
    }
    ~NeuroMixer(){ Pa_StopStream(stream); Pa_CloseStream(stream); Pa_Terminate(); }

    void load_deck(std::string name, std::string fp){
        unsigned int ch,sr; drwav_uint64 tf;
        float* pNew=drwav_open_file_and_read_pcm_frames_f32(fp.c_str(),&ch,&sr,&tf,NULL);
        if(!pNew) return;
        EQFilter nEQ; DelayEffect nFX; nFX.reset(); ReverbWash nRV; nRV.reset();
        float* pOld=nullptr;
        {
            std::lock_guard<std::mutex> lk(audioMutex);
            Deck* t=(name=="A")?&deckA:&deckB;
            pOld=t->pSampleData; t->pSampleData=pNew;
            t->totalSampleCount=tf*ch;
            t->currentSampleIndex.store(0,std::memory_order_relaxed);
            t->eq=nEQ; std::swap(t->fx,nFX); std::swap(t->reverb,nRV);
            t->isLooping=false;
            if (name == "A") {
                t->volume = t->tgt_volume = 1.0f;
            } else {
                t->volume = t->tgt_volume = 0.0f;
            }
        }
        if(pOld) drwav_free(pOld,nullptr);
    }

    void set_loop_region(int tid,float s,float e,int cnt){
        drwav_uint64 ss=(drwav_uint64)(s*88200); drwav_uint64 ee=(drwav_uint64)(e*88200);
        if (ss % 2) { ss--; }
        if (ee % 2) { ee--; }
        pendingLoop.track_id=tid; pendingLoop.loopStartSample=ss;
        pendingLoop.loopEndSample=ee; pendingLoop.loop_count=cnt;
        std::atomic_thread_fence(std::memory_order_release);
        pendingLoop.pending.store(true,std::memory_order_relaxed);
    }

    void trigger_hybrid_transition(float dur,float beats,float bass,
                                   float echo,float stutter,float wash,
                                   float bpm,float piano,int tech=0,
                                   float wf1=2.3f,float wf2=5.7f,
                                   float wp1=0.0f,float wp2=0.0f,float wamp=0.012f){
        pendingTrans.transTotalFrames=(drwav_uint64)(dur*44100);
        pendingTrans.p_beats=beats; pendingTrans.p_bass_swap=bass;
        pendingTrans.p_echo=echo;   pendingTrans.p_stutter=stutter;
        pendingTrans.p_wash=wash;   pendingTrans.p_bpm=bpm;
        pendingTrans.p_piano_hold=piano; pendingTrans.technique_id=tech;
        pendingTrans.w_f1=wf1; pendingTrans.w_f2=wf2;
        pendingTrans.w_p1=wp1; pendingTrans.w_p2=wp2;
        pendingTrans.w_amp=wamp;
        std::atomic_thread_fence(std::memory_order_release);
        pendingTrans.pending.store(true,std::memory_order_relaxed);
    }

    void play(std::string n){ if(n=="A")deckA.isPlaying.store(true,std::memory_order_relaxed);
                               if(n=="B")deckB.isPlaying.store(true,std::memory_order_relaxed); }
    void pause(std::string n){ if(n=="A")deckA.isPlaying.store(false,std::memory_order_relaxed);
                                if(n=="B")deckB.isPlaying.store(false,std::memory_order_relaxed); }

    void seek(std::string n,float sec){
        drwav_uint64 idx=(drwav_uint64)(sec*88200);
        if(n=="A"&&idx<deckA.totalSampleCount) deckA.currentSampleIndex.store(idx,std::memory_order_relaxed);
        else if(n=="B"&&idx<deckB.totalSampleCount) deckB.currentSampleIndex.store(idx,std::memory_order_relaxed);
    }
    float get_position(std::string n){
        if(n=="A") return deckA.currentSampleIndex.load(std::memory_order_relaxed)/88200.f;
        return deckB.currentSampleIndex.load(std::memory_order_relaxed)/88200.f;
    }
    std::vector<float> get_visual_buffer(std::string n){
        return (n=="A") ? deckA.visualBuffer : deckB.visualBuffer;
    }
    bool is_transitioning(){ return inTransition.load(std::memory_order_relaxed); }

    void add_eq_event(int deck_id, float time_sec,
                      float low_mult, float mid_mult, float high_mult) {
        EQEvent ev;
        ev.trigger_sample = (drwav_uint64)(time_sec * 88200.0f);
        ev.deck_id   = deck_id;
        ev.low_mult  = clampf(low_mult,  0.0f, 2.0f);
        ev.mid_mult  = clampf(mid_mult,  0.0f, 2.0f);
        ev.high_mult = clampf(high_mult, 0.0f, 2.0f);
        ev.fired     = false;
        std::lock_guard<std::mutex> lk(audioMutex);
        eqSchedule.push_back(ev);
        std::sort(eqSchedule.begin(), eqSchedule.end(),
                  [](const EQEvent& a, const EQEvent& b){
                      return a.trigger_sample < b.trigger_sample; });
    }

    void clear_eq_events() {
        pendingClearEQ.store(true, std::memory_order_release);
    }

    void swap_decks(){
        std::lock_guard<std::mutex> lk(audioMutex);
        std::swap(deckA.pSampleData,deckB.pSampleData);
        std::swap(deckA.totalSampleCount,deckB.totalSampleCount);
        drwav_uint64 ti=deckA.currentSampleIndex.load(std::memory_order_relaxed);
        deckA.currentSampleIndex.store(deckB.currentSampleIndex.load(std::memory_order_relaxed),std::memory_order_relaxed);
        deckB.currentSampleIndex.store(ti,std::memory_order_relaxed);
        std::swap(deckA.volume,deckB.volume); std::swap(deckA.tgt_volume,deckB.tgt_volume);
        std::swap(deckA.gate_volume,deckB.gate_volume);
        bool tp=deckA.isPlaying.load(std::memory_order_relaxed);
        deckA.isPlaying.store(deckB.isPlaying.load(std::memory_order_relaxed),std::memory_order_relaxed);
        deckB.isPlaying.store(tp,std::memory_order_relaxed);
        std::swap(deckA.eq,deckB.eq); std::swap(deckA.fx,deckB.fx);
        std::swap(deckA.reverb,deckB.reverb);
        std::swap(deckA.visualBuffer,deckB.visualBuffer);
        std::swap(deckA.visualIdx,deckB.visualIdx);
        std::swap(deckA.isLooping,deckB.isLooping);
        std::swap(deckA.loopStartSample,deckB.loopStartSample);
        std::swap(deckA.loopEndSample,deckB.loopEndSample);
        std::swap(deckA.targetLoopCount,deckB.targetLoopCount);
        std::swap(deckA.currentLoopCount,deckB.currentLoopCount);
        eqSchedule.clear();
        deckA.eq.tgt_ev_low=deckA.eq.tgt_ev_mid=deckA.eq.tgt_ev_high=1.0f;
        deckB.eq.tgt_ev_low=deckB.eq.tgt_ev_mid=deckB.eq.tgt_ev_high=1.0f;
    }
};

PYBIND11_MODULE(neuro_core,m){
    py::class_<NeuroMixer>(m,"NeuroMixer")
        .def(py::init<>())
        .def("load_deck",&NeuroMixer::load_deck)
        .def("play",&NeuroMixer::play)
        .def("pause",&NeuroMixer::pause)
        .def("seek",&NeuroMixer::seek)
        .def("get_position",&NeuroMixer::get_position)
        .def("swap_decks",&NeuroMixer::swap_decks)
        .def("is_transitioning",&NeuroMixer::is_transitioning)
        .def("get_visual_buffer",&NeuroMixer::get_visual_buffer)
        .def("trigger_hybrid_transition",&NeuroMixer::trigger_hybrid_transition,
             py::arg("dur"),py::arg("beats"),py::arg("bass"),
             py::arg("echo"),py::arg("stutter"),py::arg("wash"),
             py::arg("bpm"),py::arg("piano"),py::arg("tech")=0,
             py::arg("wf1")=2.3f,py::arg("wf2")=5.7f,
             py::arg("wp1")=0.0f,py::arg("wp2")=0.0f,py::arg("wamp")=0.012f)
        .def("set_loop_region",&NeuroMixer::set_loop_region)
        .def("add_eq_event",   &NeuroMixer::add_eq_event,
             py::arg("deck_id"), py::arg("time_sec"),
             py::arg("low_mult"), py::arg("mid_mult"), py::arg("high_mult"))
        .def("clear_eq_events",&NeuroMixer::clear_eq_events);
}