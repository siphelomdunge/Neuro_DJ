/*
 * core_audio.cpp  —  Neuro-DJ Audio Engine v6.2 (Complete with All Bindings)
 *
 * COMPILED WITH: Pybind11 + PortAudio + dr_wav
 *
 * V6.2 CRITICAL FIXES:
 * • Fixed race condition in transition pickup detection
 * • Added transitionConfirmed flag for reliable Python monitoring
 * • Strengthened memory ordering on critical transition paths
 * • Added get_volume() / set_volume() for multi-signal detection
 * • Added diagnostic callback heartbeat for debugging
 * • Fixed transition state never being visible to Python thread
 * • FIXED: All Python bindings now properly exposed
 *
 * V6.1 PRODUCTION FIXES:
 * • Fixed race condition: Made isLooping and currentLoopCount atomic
 * • Added bounds checking after loop index reset
 * • Protected swap_decks() against null pointer scenarios
 * • Improved visual buffer lock documentation
 * • Added const-ref optimization for string parameters
 * • Documented magic numbers for audio engineers
 * • Added denormal protection to sigmoid crossfade
 * • Cache-aligned hot-path variables in Deck structure
 * • Reset filter state on deck swap to prevent artifacts
 * • Added diagnostic lock failure counter
 *
 * V6.0 ARCHITECTURAL UPGRADES:
 * • Denormal Protection: Added zap_denorm() to Biquad, EQ, and Delay feedback 
 * paths to prevent CPU microcode spikes on near-zero floats.
 * • True RMS Math: Fixed Acapella gain compensation to use correct 2D energy averaging.
 * • NaN/Inf Master Bus Protection: Hard clamps before the soft-clipper to prevent driver crashes.
 * • Click-Free Transport: isPlaying state now triggers a fast 10ms volume slew 
 * instead of a hard mathematical discontinuity.
 * • Clean Control Flow: Eradicated goto statements in the transition engine.
 * • Thread-Safe Visuals: Double-buffered visual state for Python GUI reads.
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

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace py = pybind11;
static const float kPI = 3.14159265359f;

// ── V6.0: Denormal & Safety Utilities ─────────────────────────────────────────
inline float zap_denorm(float v) {
    // Fast absolute value check. Flushes floats smaller than 1e-15 to absolute zero.
    return (fabsf(v) < 1e-15f) ? 0.0f : v;
}

inline float safe_float(float v) {
    return std::isfinite(v) ? v : 0.0f;
}

// ── Curve library ─────────────────────────────────────────────────────────────
inline float curve_smooth(float t)  { return t * t * (3.0f - 2.0f * t); }
inline float curve_enter(float t)   { return sinf(t * (kPI / 2.0f)); }
inline float curve_exit(float t)    { return cosf(t * (kPI / 2.0f)); }

inline float sigmoid_swap(float t, float s=0.5f, float k=70.0f){
    float x = -k * (t - s);
    if(x >  80.0f) return 0.0f;
    if(x < -80.0f) return 1.0f;
    // V6.1: Added denormal protection to crossfade curves
    return zap_denorm(1.0f / (1.0f + expf(x)));
}

inline float bass_cp_a(float sigma){ return cosf(sigma * (kPI / 2.0f)); }
inline float bass_cp_b(float sigma){ return sinf(sigma * (kPI / 2.0f)); }
inline float clampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }

inline void compensated_pair(float exit_t, float entry_t, float& a_out, float& b_out){
    float a = cosf(exit_t  * (kPI / 2.0f));
    float b = sinf(entry_t * (kPI / 2.0f));
    float power = a*a + b*b;
    float comp  = (power > 0.0f) ? (1.0f / sqrtf(power)) : 1.0f;
    if(comp > 1.414f) comp = 1.414f;
    a_out = a * comp;
    b_out = b * comp;
}

inline float wobble_scaled(float gain, float t, float scale,
                            float wf1, float wf2, float wp1, float wp2, float wamp){
    if(scale < 0.01f) return clampf(gain, 0.0f, 1.0f);
    float w = wamp * scale * sinf(wf1*kPI*t + wp1) * sinf(wf2*kPI*t + wp2);
    return clampf(gain + w, 0.0f, 1.0f);
}

// Transparent soft-limiter with hard absolute ceiling
inline float soft_limit(float x){
    constexpr float D = 0.55f;  // Distortion factor (lower = harder knee)
    float y = tanhf(x * D) / D;
    return (y > 1.0f) ? 1.0f : (y < -1.0f) ? -1.0f : y;
}

// ── BiquadHPF (For Mid/Side Acapella Extraction) ─────────────────────────────
struct BiquadHPF {
    // 200 Hz Butterworth highpass @ 44.1 kHz (pre-computed coefficients)
    static constexpr float B0 =  0.974864f;
    static constexpr float B1 = -1.949727f;
    static constexpr float B2 =  0.974864f;
    static constexpr float A1 = -1.949277f;
    static constexpr float A2 =  0.950178f;
    
    float s1 = 0.0f;
    float s2 = 0.0f;
    
    inline float process(float x) {
        float y = B0 * x + s1;
        // V6.0: Prevent denormal CPU spikes
        s1 = zap_denorm(B1 * x - A1 * y + s2);
        s2 = zap_denorm(B2 * x - A2 * y);
        return y;
    }
    void reset() { s1 = s2 = 0.0f; }
};

// ── EQFilter ──────────────────────────────────────────────────────────────────
struct EQFilter {
    float low_gain=1,mid_gain=1,high_gain=1;
    float tgt_low =1,tgt_mid =1,tgt_high =1;
    float ev_low_mult=1,ev_mid_mult=1,ev_high_mult=1;
    float tgt_ev_low =1,tgt_ev_mid =1,tgt_ev_high =1;
    float l_low=0,l_high=0,r_low=0,r_high=0;
    float alpha_low, alpha_high;
    
    // V6.1: Documented slew rates for audio engineers
    static constexpr float SLEW    = 0.012f;  // ~83ms smoothing time at 44.1kHz
    static constexpr float EV_SLEW = 0.010f;  // ~100ms smoothing for event-triggered EQ

    EQFilter(float lF=250.f, float hF=4000.f){
        alpha_low  = (2*kPI*lF)  / (44100.f + 2*kPI*lF);
        alpha_high = (2*kPI*hF)  / (44100.f + 2*kPI*hF);
    }
    inline void advance(){
        low_gain  += SLEW*(tgt_low  - low_gain);
        mid_gain  += SLEW*(tgt_mid  - mid_gain);
        high_gain += SLEW*(tgt_high - high_gain);
        ev_low_mult  += EV_SLEW*(tgt_ev_low  - ev_low_mult);
        ev_mid_mult  += EV_SLEW*(tgt_ev_mid  - ev_mid_mult);
        ev_high_mult += EV_SLEW*(tgt_ev_high - ev_high_mult);
    }
    inline float process(float s, bool isL){
        float lp, hp, mp;
        if(isL){ 
            l_low+=alpha_low*(s-l_low); l_low = zap_denorm(l_low); lp=l_low; 
            l_high+=alpha_high*(s-l_high); l_high = zap_denorm(l_high); hp=s-l_high; 
        } else { 
            r_low+=alpha_low*(s-r_low); r_low = zap_denorm(r_low); lp=r_low; 
            r_high+=alpha_high*(s-r_high); r_high = zap_denorm(r_high); hp=s-r_high; 
        }
        mp = s - lp - hp;
        return lp*(low_gain*ev_low_mult) + mp*(mid_gain*ev_mid_mult) + hp*(high_gain*ev_high_mult);
    }
    void reset_flat(){
        tgt_low=tgt_mid=tgt_high=1; low_gain=mid_gain=high_gain=1;
        tgt_ev_low=tgt_ev_mid=tgt_ev_high=1; ev_low_mult=ev_mid_mult=ev_high_mult=1;
        l_low=l_high=r_low=r_high=0;
    }
};

// ── Delay ─────────────────────────────────────────────────────────────────────
struct DelayEffect {
    std::vector<float> bufL,bufR;
    int bufSize=88200, writeIdx=0, delaySamples=20000;
    float feedback=0.6f, tgt_feedback=0.6f;
    
    static constexpr float FB_SLEW = 0.002f;  // ~500ms feedback smoothing at 44.1kHz
    static constexpr float DC_FILTER_COEFF = 0.995f;  // 1-pole DC blocker (-3dB @ 3.5 Hz)
    
    float mix=0;
    
    float prevL_in=0, prevL_out=0, prevR_in=0, prevR_out=0;

    DelayEffect(){ bufL.resize(bufSize,0); bufR.resize(bufSize,0); }
    void reset(){
        std::fill(bufL.begin(),bufL.end(),0);
        std::fill(bufR.begin(),bufR.end(),0);
        prevL_in=prevL_out=prevR_in=prevR_out=0; 
        mix=0; feedback=tgt_feedback=0.6f;
    }
    inline void advance(){ feedback += FB_SLEW*(tgt_feedback - feedback); }
    inline void process(float& sL, float& sR){
        if(mix < 0.001f) return;
        int ri = writeIdx - delaySamples;
        if(ri < 0) ri += bufSize;
        float dL=bufL[ri], dR=bufR[ri];
        
        // V6.0: Denormal flush in the infinite feedback loop
        float dcL_out = zap_denorm(dL - prevL_in + DC_FILTER_COEFF * prevL_out);
        prevL_in = dL; 
        prevL_out = dcL_out;
        
        float dcR_out = zap_denorm(dR - prevR_in + DC_FILTER_COEFF * prevR_out);
        prevR_in = dR; 
        prevR_out = dcR_out;

        bufL[writeIdx] = sL + dcL_out * feedback;
        bufR[writeIdx] = sR + dcR_out * feedback;
        
        writeIdx=(writeIdx+1)%bufSize;
        sL=sL*(1-mix)+dL*mix;
        sR=sR*(1-mix)+dR*mix;
    }
};

// ── Reverb ────────────────────────────────────────────────────────────────────
struct ReverbWash {
    static const int NC=4;
    static const int D[NC];  // Prime-number delay lines for natural wash
    float bL[NC][1400], bR[NC][1400];
    int   ix[NC];
    float fb=0.76f, mix=0;
    ReverbWash(){ reset(); }
    void reset(){ 
        for(int i=0; i<NC; i++) {
            std::fill(bL[i], bL[i]+1400, 0.0f);
            std::fill(bR[i], bR[i]+1400, 0.0f);
            ix[i] = 0;
        }
        mix=0; 
    }
    inline void process(float& L, float& R){
        if(mix < 0.001f) return;
        float wL=0, wR=0;
        for(int i=0;i<NC;i++){
            float oL=bL[i][ix[i]], oR=bR[i][ix[i]];
            // V6.0: Denormal protection in reverb comb lines
            bL[i][ix[i]] = zap_denorm(L+oL*fb); 
            bR[i][ix[i]] = zap_denorm(R+oR*fb);
            ix[i]=(ix[i]+1)%D[i];
            wL+=oL; wR+=oR;
        }
        wL/=NC; wR/=NC;
        L=L*(1-mix)+wL*mix; R=R*(1-mix)+wR*mix;
    }
};
const int ReverbWash::D[4]={1031,1153,1237,1361};

// ── Deck ──────────────────────────────────────────────────────────────────────
struct Deck {
    // V6.1: Cache-aligned hot path variables (grouped for CPU cache efficiency)
    alignas(64) std::atomic<drwav_uint64> currentSampleIndex{0};
    std::atomic<bool> isPlaying{false};
    float transport_gate = 0.0f;
    float volume=1, tgt_volume=1;
    float gate_volume=1.0f;
    
    static constexpr float TRANSPORT_SLEW = 0.002f;  // ~10ms click-free ramp at 44.1kHz
    static constexpr float VOL_SLEW = 0.008f;        // ~125ms volume smoothing

    // Cold path: track metadata (rarely accessed in audio callback)
    float* pSampleData      = nullptr;
    drwav_uint64    totalSampleCount = 0;
    unsigned int    channels         = 2;

    EQFilter   eq;
    DelayEffect fx;
    ReverbWash  reverb;

    // Acapella State
    float acapellaIntensity = 0.0f;
    float acapellaCurrent = 0.0f;
    static constexpr float ACAPELLA_SMOOTH = 0.0005f;  // ~2 second blend time
    BiquadHPF midHPF;

    // V6.1: Thread-Safe Visual Buffer
    std::vector<float> visualBufferAudioThread;
    std::vector<float> visualBufferPythonThread;
    std::atomic_flag visualBufferLock = ATOMIC_FLAG_INIT;
    int visualIdx = 0;

    // V6.1: Made loop state atomic for thread safety
    std::atomic<bool> isLooping{false};
    drwav_uint64 loopStartSample=0, loopEndSample=0;
    int targetLoopCount=0;
    std::atomic<int> currentLoopCount{0};

    float tail_fade_gain = 1.0f;
    bool  in_tail_fade   = false;
    static constexpr float TAIL_FADE_SLEW = 1.0f / (4.0f * 44100.0f);  // 4 second fade

    Deck(){ 
        visualBufferAudioThread.resize(512, 0); 
        visualBufferPythonThread.resize(512, 0);
    }
    ~Deck(){ if(pSampleData){ drwav_free(pSampleData, nullptr); pSampleData=nullptr; } }

    inline void advance(){
        volume += VOL_SLEW*(tgt_volume - volume);
        
        // V6.0: Click-free transport slew
        float target_transport = isPlaying.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        transport_gate += TRANSPORT_SLEW * (target_transport - transport_gate);
        
        eq.advance();
        fx.advance();
        if(in_tail_fade){
            tail_fade_gain -= TAIL_FADE_SLEW;
            if(tail_fade_gain <= 0.0f){
                tail_fade_gain = 0.0f;
                tgt_volume     = 0.0f;
                in_tail_fade   = false;
            }
        }
    }
    
    void reset_state(){
        eq.reset_flat(); fx.reset(); reverb.reset();
        volume=tgt_volume=1; gate_volume=1; 
        isLooping.store(false, std::memory_order_relaxed);
        tail_fade_gain=1.0f; in_tail_fade=false;
        transport_gate = 0.0f; // Hard reset on load
        acapellaIntensity = 0.0f; acapellaCurrent = 0.0f; midHPF.reset();
        currentLoopCount.store(0, std::memory_order_relaxed);
    }

    inline drwav_uint64 stride() const {
        // Returns samples per second for this track (sample_rate * channels)
        return static_cast<drwav_uint64>(44100) * std::max(1u, channels);
    }
};

// ── Acapella Isolation processing ─────────────────────────────────────────────
inline void applyAcapellaExtraction(Deck& d, float& L, float& R) {
    float target = d.acapellaIntensity;
    
    float error = target - d.acapellaCurrent;
    d.acapellaCurrent += clampf(error, -Deck::ACAPELLA_SMOOTH, Deck::ACAPELLA_SMOOTH);
    
    float intensity = d.acapellaCurrent;
    
    if (intensity < 0.001f) return;
    
    // 1. Encode to Mid/Side
    float mid  = (L + R) * 0.5f;
    float side = (L - R) * 0.5f;
    
    // 2. Gentle side reduction (stereo width narrowing for vocal focus)
    float sideAttenuation = 0.15f + 0.85f * (1.0f - intensity);
    float sideReduced = side * sideAttenuation;
    
    // 3. High-pass mid (removes low-freq instruments, preserves vocals)
    float midFiltered = d.midHPF.process(mid);
    float midBlended = mid * (1.0f - intensity) + midFiltered * intensity;
    
    // 4. V6.0: True RMS computation (correct 2D energy preservation)
    float inputRMS  = sqrtf(0.5f * (mid*mid + side*side));
    float outputRMS = sqrtf(0.5f * (midBlended*midBlended + sideReduced*sideReduced));
    
    float compensation = (outputRMS > 0.0001f) 
        ? clampf(inputRMS / outputRMS, 0.85f, 1.25f) 
        : 1.0f;
    
    // 5. Decode back to L/R
    L = (midBlended + sideReduced) * compensation;
    R = (midBlended - sideReduced) * compensation;
}

// ── Scheduled EQ event ────────────────────────────────────────────────────────
struct EQEvent {
    drwav_uint64 trigger_sample;
    int   deck_id;
    float low_mult, mid_mult, high_mult;
    bool  fired = false;
};

// ── Lock-free command queues ──────────────────────────────────────────────────
struct LoopCommand {
    std::atomic<bool> pending{false};
    int track_id;
    drwav_uint64 loopStartSample, loopEndSample;
    int loop_count;
};

struct TransCommand {
    std::atomic<bool> pending{false};
    drwav_uint64 transTotalFrames;
    float p_beats,p_bass_swap,p_echo,p_stutter,p_wash,p_bpm,p_piano_hold;
    int   technique_id;
    float w_f1=2.3f, w_f2=5.7f, w_p1=0.0f, w_p2=0.0f, w_amp=0.012f;
};

// ── NeuroMixer ────────────────────────────────────────────────────────────────
class NeuroMixer {
    Deck deckA, deckB;
    PaStream* stream;
    std::mutex audioMutex;
    
    // V6.0: Lock-fail safety buffer
    float lastOutL = 0.0f;
    float lastOutR = 0.0f;
    
    // V6.1: Diagnostic counter for lock failures
    std::atomic<uint64_t> lockFailureCount{0};

    std::atomic<bool> inTransition{false};
    
    // V6.2: Critical fix - confirmation flag that persists for Python detection
    std::atomic<bool> transitionConfirmed{false};
    
    drwav_uint64 transTotalFrames=0, transCurrentFrame=0;
    float p_beats=16, p_bass_swap=0.75f, p_echo=0, p_stutter=0,
          p_wash=0, p_bpm=112, p_piano_hold=0;
    int technique_id=0;
    float w_f1=2.3f, w_f2=5.7f, w_p1=0.0f, w_p2=0.0f, w_amp=0.012f;

    LoopCommand  pendingLoop;
    TransCommand pendingTrans;

    std::vector<EQEvent>  eqSchedule;
    std::atomic<bool>     pendingClearEQ{false};

    // V6.2: Diagnostic heartbeat
    std::atomic<uint64_t> callbackHeartbeat{0};

    // ── Audio Callback ────────────────────────────────────────────────────────
    static int audioCallback(const void*, void* outBuf, unsigned long nFrames,
                             const PaStreamCallbackTimeInfo*,
                             PaStreamCallbackFlags, void* ud){
        NeuroMixer* mx = (NeuroMixer*)ud;
        float* out     = (float*)outBuf;
        
        // V6.2: Heartbeat for diagnostics (increment every callback)
        mx->callbackHeartbeat.fetch_add(1, std::memory_order_relaxed);

        // V6.1: Process loop command (lock-free)
        if(mx->pendingLoop.pending.load(std::memory_order_acquire)){
            Deck* t = (mx->pendingLoop.track_id==0) ? &mx->deckA : &mx->deckB;
            t->loopStartSample  = mx->pendingLoop.loopStartSample;
            t->loopEndSample    = mx->pendingLoop.loopEndSample;
            t->targetLoopCount  = mx->pendingLoop.loop_count;
            t->currentLoopCount.store(0, std::memory_order_relaxed);
            t->isLooping.store(mx->pendingLoop.loop_count > 0, std::memory_order_relaxed);
            mx->pendingLoop.pending.store(false, std::memory_order_release);
        }

        // V6.2: Critical fix - strengthened memory ordering for transition handoff
        if(mx->pendingTrans.pending.load(std::memory_order_acquire)){
            mx->transTotalFrames  = mx->pendingTrans.transTotalFrames;
            mx->transCurrentFrame = 0;
            mx->p_beats           = mx->pendingTrans.p_beats;
            mx->p_bass_swap       = mx->pendingTrans.p_bass_swap;
            mx->p_echo            = mx->pendingTrans.p_echo;
            mx->p_stutter         = mx->pendingTrans.p_stutter;
            mx->p_wash            = mx->pendingTrans.p_wash;
            mx->p_bpm             = mx->pendingTrans.p_bpm;
            mx->p_piano_hold      = mx->pendingTrans.p_piano_hold;
            mx->technique_id      = mx->pendingTrans.technique_id;
            mx->w_f1=mx->pendingTrans.w_f1; mx->w_f2=mx->pendingTrans.w_f2;
            mx->w_p1=mx->pendingTrans.w_p1; mx->w_p2=mx->pendingTrans.w_p2;
            mx->w_amp=mx->pendingTrans.w_amp;

            mx->deckA.fx.reset(); mx->deckA.reverb.reset();
            mx->deckB.fx.reset(); mx->deckB.reverb.reset();
            mx->deckA.eq.tgt_low=mx->deckA.eq.tgt_mid=mx->deckA.eq.tgt_high=1;
            mx->deckA.eq.low_gain=mx->deckA.eq.mid_gain=mx->deckA.eq.high_gain=1;
            mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1;
            mx->deckB.eq.low_gain=mx->deckB.eq.mid_gain=mx->deckB.eq.high_gain=1;
            mx->deckA.volume=1.0f; mx->deckA.tgt_volume=1.0f; mx->deckA.gate_volume=1.0f;
            mx->deckB.volume=0.0f; mx->deckB.tgt_volume=0.0f; mx->deckB.gate_volume=1.0f;
            mx->deckA.tail_fade_gain=1.0f; mx->deckA.in_tail_fade=false;
            mx->deckB.tail_fade_gain=1.0f; mx->deckB.in_tail_fade=false;
            mx->deckB.isPlaying.store(true, std::memory_order_relaxed);
            
            // V6.2: Set BOTH flags with strong memory ordering
            mx->inTransition.store(true, std::memory_order_seq_cst);
            mx->transitionConfirmed.store(true, std::memory_order_seq_cst);
            
            mx->pendingTrans.pending.store(false, std::memory_order_release);
        }

        std::unique_lock<std::mutex> lock(mx->audioMutex, std::try_to_lock);
        
        // V6.1: Output last safe sample on lock failure + increment diagnostic counter
        if(!lock.owns_lock()){
            mx->lockFailureCount.fetch_add(1, std::memory_order_relaxed);
            for(unsigned int i = 0; i < nFrames; i++) {
                *out++ = mx->lastOutL;
                *out++ = mx->lastOutR;
            }
            return paContinue;
        }

        bool isTrans = mx->inTransition.load(std::memory_order_relaxed);

        if(mx->pendingClearEQ.load(std::memory_order_relaxed)){
            mx->eqSchedule.clear();
            mx->deckA.eq.tgt_ev_low=mx->deckA.eq.tgt_ev_mid=mx->deckA.eq.tgt_ev_high=1.0f;
            mx->deckB.eq.tgt_ev_low=mx->deckB.eq.tgt_ev_mid=mx->deckB.eq.tgt_ev_high=1.0f;
            mx->pendingClearEQ.store(false, std::memory_order_relaxed);
        }

        if(!mx->eqSchedule.empty()){
            drwav_uint64 posA = mx->deckA.currentSampleIndex.load(std::memory_order_relaxed);
            drwav_uint64 posB = mx->deckB.currentSampleIndex.load(std::memory_order_relaxed);
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

        for(unsigned int i = 0; i < nFrames; i++){

            mx->deckA.advance();
            mx->deckB.advance();
            
            if(isTrans){
                mx->transCurrentFrame++;

                if(mx->transTotalFrames == 0){
                    // Instant cut
                    mx->inTransition.store(false, std::memory_order_seq_cst);
                    mx->transitionConfirmed.store(false, std::memory_order_seq_cst);
                    isTrans = false;
                    mx->deckA.tgt_volume=0; mx->deckB.tgt_volume=1;
                    mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1;
                    mx->deckA.fx.mix=0; mx->deckA.reverb.mix=0; mx->deckB.reverb.mix=0;
                    mx->deckA.gate_volume=1.0f; mx->deckB.gate_volume=1.0f;
                } else {
                    float prog = (float)mx->transCurrentFrame / (float)mx->transTotalFrames;

                    if(prog >= 1.f){
                        // V6.2: Transition complete - clear confirmation flag
                        mx->inTransition.store(false, std::memory_order_seq_cst);
                        mx->transitionConfirmed.store(false, std::memory_order_seq_cst);
                        isTrans = false;
                        mx->deckB.tgt_volume=1.0f;
                        mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1.0f;
                        mx->deckA.reverb.mix=0; mx->deckB.reverb.mix=0;
                        mx->deckA.gate_volume=1.0f; mx->deckB.gate_volume=1.0f;

                        if(mx->technique_id == 3 || mx->technique_id == 7){
                            mx->deckA.tgt_volume  = 0.4f;
                            mx->deckA.eq.tgt_low  = 0.0f;
                            mx->deckA.in_tail_fade = true;
                            mx->deckA.tail_fade_gain = 0.4f;
                        } else {
                            mx->deckA.tgt_volume = 0.0f;
                            mx->deckA.fx.mix     = 0.0f;
                        }
                    } else {
                        // Transition active logic
                        float beat_samples  = 44100.0f * 60.0f / mx->p_bpm;
                        float beat_t        = (float)mx->transCurrentFrame / beat_samples;
                        float total_beats   = mx->p_beats;

                        float safe_bass_swap = mx->p_bass_swap;
                        if(mx->technique_id==0||mx->technique_id==2||mx->technique_id==4)
                            safe_bass_swap = clampf(mx->p_bass_swap, 0.40f, 0.75f);
                        else if(mx->technique_id==5)
                            safe_bass_swap = clampf(mx->p_bass_swap, 0.40f, 0.55f);

                        auto wobble = [&](float gain, float t, float scale=1.0f) -> float {
                            return wobble_scaled(gain, t, scale,
                                mx->w_f1, mx->w_f2, mx->w_p1, mx->w_p2, mx->w_amp);
                        };

                        switch(mx->technique_id){

                        case 0: default: {  // Smooth Crossfade (default technique)
                            float swap_beat  = total_beats * safe_bass_swap;
                            float t_all      = clampf(beat_t / total_beats, 0.0f, 1.0f);
                            float t_phase1   = clampf(beat_t / clampf(swap_beat,1.f,9999.f), 0.0f,1.0f);
                            float bass_sigma = sigmoid_swap(t_all, safe_bass_swap, 70.0f);
                            float t_fade     = clampf((beat_t - swap_beat - 2.0f)
                                                     / clampf(total_beats - swap_beat - 2.0f, 1.f, 9999.f),
                                                     0.0f, 1.0f);

                            mx->deckA.eq.tgt_low = bass_cp_a(bass_sigma);
                            mx->deckB.eq.tgt_low = bass_cp_b(bass_sigma);

                            float b_mid_t  = clampf((t_phase1 - 0.40f) / 0.60f, 0.0f, 1.0f);
                            float b_high_t = t_phase1;

                            if(beat_t < swap_beat){
                                mx->deckA.tgt_volume  = 1.0f;
                                mx->deckA.eq.tgt_mid  = 1.0f;
                                mx->deckA.eq.tgt_high = 1.0f;
                                float _am,_bm,_ah,_bh;
                                compensated_pair(0.0f, b_mid_t,  _am, _bm);
                                compensated_pair(0.0f, b_high_t, _ah, _bh);
                                mx->deckB.eq.tgt_mid  = _bm;
                                mx->deckB.eq.tgt_high = _bh;
                                float active = clampf((t_phase1 - 0.2f) / 0.8f, 0.0f, 1.0f);
                                mx->deckB.tgt_volume  = wobble(curve_enter(t_phase1)*0.85f, t_phase1, active);
                            } else {
                                float a_mid_t = clampf(t_fade / 0.85f, 0.0f, 1.0f);
                                float _am,_bm,_ah,_bh;
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

                        case 2: {  // EQ Sweep Transition
                            float t_full     = clampf(beat_t / total_beats, 0.0f, 1.0f);
                            float bass_sigma = sigmoid_swap(t_full, 0.55f, 70.0f);

                            mx->deckA.eq.tgt_low = bass_cp_a(bass_sigma);
                            mx->deckB.eq.tgt_low = bass_cp_b(bass_sigma);

                            mx->deckA.eq.tgt_high = 1.0f - curve_smooth(clampf(beat_t/(total_beats*0.65f),0.f,1.f));
                            mx->deckA.eq.tgt_mid  = 1.0f - curve_smooth(clampf((beat_t-total_beats*0.25f)/(total_beats*0.60f),0.f,1.f));
                            mx->deckB.eq.tgt_high = curve_smooth(clampf(beat_t/(total_beats*0.50f),0.f,1.f));
                            mx->deckB.eq.tgt_mid  = curve_smooth(clampf((beat_t-total_beats*0.30f)/(total_beats*0.45f),0.f,1.f));

                            mx->deckA.tgt_volume = wobble(curve_exit(t_full), t_full);
                            mx->deckB.tgt_volume = wobble(curve_enter(t_full)*0.9f, t_full);
                            mx->deckA.reverb.mix = curve_smooth(clampf((beat_t-total_beats*0.20f)/(total_beats*0.60f),0.f,1.f))
                                                  * mx->p_wash * 0.55f;
                            break;
                        }

                        case 3: {  // Echo-Out Technique
                            float trap_beat    = fminf(total_beats * 0.65f, total_beats - 5.0f);
                            float kill_beat    = fminf(trap_beat + 4.0f, total_beats - 1.0f);
                            
                            float overlap_prog = clampf(beat_t / trap_beat, 0.0f, 1.0f);
                            float trap_prog    = clampf((beat_t - trap_beat) / 4.0f, 0.0f, 1.0f);
                            bool  past_kill    = (beat_t >= kill_beat);

                            mx->deckA.fx.delaySamples = (int)beat_samples;
                            mx->deckB.fx.mix          = 0.0f;

                            if(!past_kill){
                                mx->deckA.gate_volume    = 1.0f;
                                mx->deckA.tgt_volume     = 1.0f;
                                mx->deckA.eq.tgt_low=mx->deckA.eq.tgt_mid=mx->deckA.eq.tgt_high=1.0f;
                                mx->deckA.fx.mix         = curve_smooth(trap_prog) * mx->p_echo;
                                mx->deckA.fx.tgt_feedback= 0.60f + curve_smooth(trap_prog)*0.28f;
                                mx->deckB.tgt_volume     = wobble(curve_enter(overlap_prog)*0.75f, overlap_prog);
                                mx->deckB.eq.tgt_low     = 0.0f;
                                mx->deckB.eq.tgt_mid     = 0.85f;
                                mx->deckB.eq.tgt_high    = 1.0f;
                            } else {
                                mx->deckA.gate_volume    = 0.0f;
                                mx->deckA.tgt_volume     = 0.4f;
                                mx->deckA.eq.tgt_low     = 0.0f;
                                mx->deckA.fx.mix         = 1.0f;
                                mx->deckA.fx.tgt_feedback= 0.75f;
                                mx->deckA.in_tail_fade   = true;
                                mx->deckB.tgt_volume     = 1.0f;
                                mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1.0f;
                            }
                            break;
                        }

                        case 4: {  // Simple Crossfade
                            float angle = prog * (kPI / 2.0f);
                            mx->deckA.tgt_volume = wobble(cosf(angle), prog);
                            mx->deckB.tgt_volume = wobble(sinf(angle), prog);
                            mx->deckA.eq.tgt_low=mx->deckA.eq.tgt_mid=mx->deckA.eq.tgt_high=1.0f;
                            mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1.0f;
                            break;
                        }

                        case 5: {  // Piano Roll Technique
                            float swap_beat    = total_beats * safe_bass_swap;
                            float presig_beat  = fmaxf(swap_beat - 4.0f, 0.0f);
                            float presig_dur   = fmaxf(swap_beat - presig_beat, 1.0f);
                            float farewell_dur = fmaxf(total_beats - swap_beat - 4.0f, 1.0f);

                            float bass_t       = clampf((beat_t - swap_beat) / 1.0f, 0.0f, 1.0f);
                            float piano_t      = clampf((beat_t - swap_beat - 2.0f) / 2.0f, 0.0f, 1.0f);
                            float farewell_t   = clampf((beat_t - swap_beat - 4.0f) / farewell_dur, 0.0f, 1.0f);

                            float s1_end       = fminf(16.0f, swap_beat * 0.4f);
                            float s2_dur       = fmaxf(swap_beat - s1_end, 2.0f);
                            float stage1_t     = clampf(beat_t / fmaxf(s1_end, 1.0f), 0.0f, 1.0f);
                            float stage2_t     = clampf((beat_t - s1_end) / s2_dur, 0.0f, 1.0f);
                            float presig_t     = clampf((beat_t - presig_beat) / presig_dur, 0.0f, 1.0f);

                            if(beat_t < swap_beat){
                                mx->deckA.tgt_volume  = 1.0f;
                                
                                mx->deckA.eq.tgt_low  = 1.0f - curve_smooth(presig_t) * 0.85f;
                                mx->deckA.eq.tgt_mid  = 1.0f;
                                mx->deckA.eq.tgt_high = 1.0f - presig_t * 0.15f;

                                float b_vol = curve_smooth(stage1_t) * 0.10f
                                            + curve_enter(stage2_t) * 0.70f;
                                mx->deckB.tgt_volume  = wobble(b_vol, stage2_t);
                                mx->deckB.eq.tgt_low  = 0.0f;
                                mx->deckB.eq.tgt_mid  = curve_smooth(stage2_t) * 0.35f;
                                mx->deckB.eq.tgt_high = curve_smooth(stage1_t) * 0.30f
                                                      + curve_smooth(stage2_t) * 0.70f;
                                mx->deckB.reverb.mix  = 0.0f;
                            } else {
                                float bass_sigma = sigmoid_swap(bass_t, 0.1f, 90.0f);
                                mx->deckA.eq.tgt_low  = bass_cp_a(bass_sigma) * 0.15f;

                                float accent = bass_cp_b(bass_sigma)
                                              + (1.0f - clampf(beat_t - swap_beat, 0.f, 1.f)) * 0.15f;
                                mx->deckB.eq.tgt_low  = clampf(accent, 0.0f, 1.15f);

                                float flick_rate = 4.0f;
                                float flick_osc  = 0.5f + 0.5f * sinf(
                                    beat_t * flick_rate * 2.0f * kPI);
                                float flick_env  = clampf(piano_t, 0.0f, 1.0f)
                                                 * (1.0f - clampf(beat_t - swap_beat - 3.0f, 0.0f, 1.0f));
                                float shake      = flick_env * flick_osc;

                                mx->deckA.eq.tgt_mid  = clampf(
                                    curve_exit(piano_t) + shake * 0.18f, 0.0f, 1.18f);
                                mx->deckB.eq.tgt_mid  = clampf(
                                    0.35f + 0.65f * curve_enter(piano_t) + shake * 0.18f,
                                    0.0f, 1.18f);

                                mx->deckA.eq.tgt_high = 0.85f - curve_smooth(farewell_t) * 0.85f;
                                mx->deckB.eq.tgt_high = 1.00f;

                                float b_vol = 0.80f + curve_enter(piano_t) * 0.20f;
                                mx->deckB.tgt_volume  = wobble(b_vol, piano_t);

                                mx->deckA.tgt_volume  = wobble(
                                    curve_exit(farewell_t), farewell_t);
                                
                                mx->deckA.reverb.mix  = clampf(
                                    curve_enter(farewell_t) * mx->p_wash * 0.45f, 0.0f, 0.45f);
                                mx->deckB.reverb.mix  = 0.0f;
                            }
                            break;
                        }

                        case 7: {  // Reverse Echo-Out
                            float trap_start = total_beats - 6.0f;
                            float kill_beat  = total_beats - 2.0f;
                            float trap_prog  = clampf((beat_t - trap_start) / 4.0f, 0.0f, 1.0f);
                            float kill_prog  = clampf((beat_t - kill_beat)  / 2.0f, 0.0f, 1.0f);

                            mx->deckA.fx.delaySamples = (int)beat_samples;
                            mx->deckB.fx.mix          = 0.0f;

                            if(beat_t < trap_start){
                                mx->deckA.gate_volume=1.0f; mx->deckA.tgt_volume=1.0f;
                                mx->deckA.eq.tgt_low=mx->deckA.eq.tgt_mid=mx->deckA.eq.tgt_high=1.0f;
                                mx->deckA.fx.mix=0.0f; mx->deckA.fx.tgt_feedback=0.60f;
                                mx->deckB.tgt_volume=0.0f;
                                mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=0.0f;
                            } else if(beat_t < kill_beat){
                                mx->deckA.gate_volume     =1.0f; mx->deckA.tgt_volume=1.0f;
                                mx->deckA.fx.mix          = curve_enter(trap_prog);
                                mx->deckA.fx.tgt_feedback = 0.60f + trap_prog*0.30f;
                                mx->deckB.tgt_volume      = 0.0f;
                                
                                float eq_ramp = curve_smooth(trap_prog);
                                mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=eq_ramp;
                            } else {
                                mx->deckA.gate_volume     = 0.0f;
                                mx->deckA.tgt_volume      = 0.5f;
                                mx->deckA.eq.tgt_low      = 0.0f;
                                mx->deckA.fx.mix          = 1.0f;
                                mx->deckA.fx.tgt_feedback = 0.85f;
                                mx->deckA.in_tail_fade    = true;
                                mx->deckB.gate_volume     = 1.0f;
                                mx->deckB.tgt_volume      = curve_enter(kill_prog);
                                mx->deckB.eq.tgt_low=mx->deckB.eq.tgt_mid=mx->deckB.eq.tgt_high=1.0f;
                            }
                            break;
                        }

                        case 8: {  // Bass Swap Transition
                            float swap_beat = total_beats * safe_bass_swap;
                            
                            if (beat_t < swap_beat) {
                                mx->deckA.tgt_volume = 1.0f;
                                float b_prog = clampf(beat_t / swap_beat, 0.0f, 1.0f);
                                mx->deckB.tgt_volume = curve_enter(b_prog) * 0.70f;
                                mx->deckB.eq.tgt_low = 0.0f; 
                                mx->deckB.eq.tgt_mid = curve_smooth(b_prog) * 0.85f;
                                mx->deckB.eq.tgt_high = curve_smooth(b_prog);
                            } else {
                                float post_prog = clampf((beat_t - swap_beat) / 4.0f, 0.0f, 1.0f);
                                mx->deckB.tgt_volume = 1.0f;
                                mx->deckB.eq.tgt_low = 1.0f;
                                mx->deckB.eq.tgt_mid = 1.0f;
                                mx->deckB.eq.tgt_high = 1.0f;
                                mx->deckA.tgt_volume = 1.0f - curve_smooth(post_prog);
                            }
                            break;
                        }
                        }  // end switch
                    }
                }
            } 
            
            // ── Render Deck A ────────────────────────────────────────────────
            float oLA=0, oRA=0;
            if(mx->deckA.transport_gate > 0.001f && mx->deckA.pSampleData){
                drwav_uint64 idx = mx->deckA.currentSampleIndex.load(std::memory_order_relaxed);
                drwav_uint64 step = mx->deckA.channels;
                
                // V6.1: Critical fix - Check loop bounds BEFORE reading
                bool looping = mx->deckA.isLooping.load(std::memory_order_relaxed);
                if(looping && idx >= mx->deckA.loopEndSample){
                    int loopCount = mx->deckA.currentLoopCount.fetch_add(1, std::memory_order_relaxed) + 1;
                    if(loopCount >= mx->deckA.targetLoopCount){
                        mx->deckA.isLooping.store(false, std::memory_order_relaxed);
                        // Continue playback past loop end
                    } else {
                        idx = mx->deckA.loopStartSample;
                    }
                }
                
                // V6.1: Safety bounds check after loop processing
                if(idx + step <= mx->deckA.totalSampleCount){
                    float sL = mx->deckA.pSampleData[idx];
                    float sR = (step >= 2) ? mx->deckA.pSampleData[idx+1] : sL;
                    
                    applyAcapellaExtraction(mx->deckA, sL, sR);
                    
                    oLA = mx->deckA.eq.process(sL, true)  * mx->deckA.volume * mx->deckA.gate_volume * mx->deckA.transport_gate;
                    oRA = mx->deckA.eq.process(sR, false) * mx->deckA.volume * mx->deckA.gate_volume * mx->deckA.transport_gate;
                    if(mx->deckA.in_tail_fade){ oLA *= mx->deckA.tail_fade_gain; oRA *= mx->deckA.tail_fade_gain; }
                    mx->deckA.fx.process(oLA, oRA);
                    mx->deckA.reverb.process(oLA, oRA);
                    mx->deckA.currentSampleIndex.store(idx + step, std::memory_order_relaxed);
                }
            }

            // ── Render Deck B ────────────────────────────────────────────────
            float oLB=0, oRB=0;
            if(mx->deckB.transport_gate > 0.001f && mx->deckB.pSampleData){
                drwav_uint64 idx  = mx->deckB.currentSampleIndex.load(std::memory_order_relaxed);
                drwav_uint64 step = mx->deckB.channels;
                
                // V6.1: Critical fix - Check loop bounds BEFORE reading
                bool looping = mx->deckB.isLooping.load(std::memory_order_relaxed);
                if(looping && idx >= mx->deckB.loopEndSample){
                    int loopCount = mx->deckB.currentLoopCount.fetch_add(1, std::memory_order_relaxed) + 1;
                    if(loopCount >= mx->deckB.targetLoopCount){
                        mx->deckB.isLooping.store(false, std::memory_order_relaxed);
                        // Continue playback past loop end
                    } else {
                        idx = mx->deckB.loopStartSample;
                    }
                }
                
                // V6.1: Safety bounds check after loop processing
                if(idx + step <= mx->deckB.totalSampleCount){
                    float sL = mx->deckB.pSampleData[idx];
                    float sR = (step >= 2) ? mx->deckB.pSampleData[idx+1] : sL;
                    
                    applyAcapellaExtraction(mx->deckB, sL, sR);
                    
                    oLB = mx->deckB.eq.process(sL, true)  * mx->deckB.volume * mx->deckB.gate_volume * mx->deckB.transport_gate;
                    oRB = mx->deckB.eq.process(sR, false) * mx->deckB.volume * mx->deckB.gate_volume * mx->deckB.transport_gate;
                    mx->deckB.fx.process(oLB, oRB);
                    mx->deckB.reverb.process(oLB, oRB);
                    mx->deckB.currentSampleIndex.store(idx + step, std::memory_order_relaxed);
                }
            }

            // ── Visual buffer (downsampled waveform for GUI) ─────────────────
            if(i % 4 == 0){
                mx->deckA.visualBufferAudioThread[mx->deckA.visualIdx] = oLA;
                mx->deckB.visualBufferAudioThread[mx->deckB.visualIdx] = oLB;
                
                mx->deckA.visualIdx = (mx->deckA.visualIdx + 1) % 512;
                mx->deckB.visualIdx = (mx->deckB.visualIdx + 1) % 512;
                
                // V6.1: Thread-safe buffer swap (lock-free atomic flag)
                if(mx->deckA.visualIdx == 0) {
                    if (!mx->deckA.visualBufferLock.test_and_set(std::memory_order_acquire)) {
                        mx->deckA.visualBufferPythonThread = mx->deckA.visualBufferAudioThread;
                        mx->deckA.visualBufferLock.clear(std::memory_order_release);
                    }
                    if (!mx->deckB.visualBufferLock.test_and_set(std::memory_order_acquire)) {
                        mx->deckB.visualBufferPythonThread = mx->deckB.visualBufferAudioThread;
                        mx->deckB.visualBufferLock.clear(std::memory_order_release);
                    }
                }
            }

            // ── Master bus (V6.0: NaN Protection & V6.1: Cached Fallback) ────
            float outL = safe_float(oLA + oLB);
            float outR = safe_float(oRA + oRB);
            
            mx->lastOutL = soft_limit(outL);
            mx->lastOutR = soft_limit(outR);

            *out++ = mx->lastOutL;
            *out++ = mx->lastOutR;
        }
        return paContinue;
    }

public:
    NeuroMixer(){
        PaError e = Pa_Initialize();
        if(e != paNoError) throw std::runtime_error(Pa_GetErrorText(e));
        e = Pa_OpenDefaultStream(&stream, 0, 2, paFloat32, 44100, 256, audioCallback, this);
        if(e != paNoError){ Pa_Terminate(); throw std::runtime_error(Pa_GetErrorText(e)); }
        e = Pa_StartStream(stream);
        if(e != paNoError){ Pa_CloseStream(stream); Pa_Terminate(); throw std::runtime_error(Pa_GetErrorText(e)); }
    }
    ~NeuroMixer(){ 
        Pa_StopStream(stream); 
        Pa_CloseStream(stream); 
        Pa_Terminate(); 
    }

    void load_deck(const std::string& name, const std::string& fp){
        unsigned int ch, sr;
        drwav_uint64 tf;
        float* pNew = drwav_open_file_and_read_pcm_frames_f32(fp.c_str(), &ch, &sr, &tf, NULL);
        if(!pNew) throw std::runtime_error("Failed to load WAV file: " + fp);
        if(sr != 44100){
            drwav_free(pNew, nullptr);
            throw std::runtime_error("Unsupported sample rate: " + std::to_string(sr) + ". Engine requires 44100 Hz.");
        }

        EQFilter nEQ;
        DelayEffect nFX; nFX.reset();
        ReverbWash nRV;  nRV.reset();
        float* pOld = nullptr;
        {
            std::lock_guard<std::mutex> lk(audioMutex);
            Deck* t = (name=="A") ? &deckA : &deckB;
            pOld = t->pSampleData;
            t->pSampleData       = pNew;
            t->totalSampleCount  = tf * ch;
            t->channels          = ch;
            t->currentSampleIndex.store(0, std::memory_order_relaxed);
            t->eq = nEQ;
            std::swap(t->fx,     nFX);
            std::swap(t->reverb, nRV);
            t->reset_state();
            t->volume            = (name=="A") ? 1.0f : 0.0f;
            t->tgt_volume        = (name=="A") ? 1.0f : 0.0f;
            t->isPlaying.store(false, std::memory_order_relaxed);
        }
        if(pOld) drwav_free(pOld, nullptr);
    }

    void set_loop_region(int tid, float s, float e, int cnt){
        Deck* t = (tid==0) ? &deckA : &deckB;
        drwav_uint64 strideVal = t->stride();
        drwav_uint64 ss = (drwav_uint64)(s * strideVal);
        drwav_uint64 ee = (drwav_uint64)(e * strideVal);
        
        // V6.1: Align to channel boundaries for stereo tracks
        if(t->channels >= 2){
            if(ss % 2) ss--;
            if(ee % 2) ee--;
        }
        
        // V6.1: Bounds safety
        if(ss >= t->totalSampleCount) ss = 0;
        if(ee > t->totalSampleCount) ee = t->totalSampleCount;
        if(ss >= ee) return;  // Invalid loop region
        
        pendingLoop.track_id        = tid;
        pendingLoop.loopStartSample = ss;
        pendingLoop.loopEndSample   = ee;
        pendingLoop.loop_count      = cnt;
        std::atomic_thread_fence(std::memory_order_release);
        pendingLoop.pending.store(true, std::memory_order_relaxed);
    }

    // V6.2: FIXED - Now properly exposed to Python
    void trigger_hybrid_transition(float dur, float beats, float bass,
                                   float echo, float stutter, float wash,
                                   float bpm, float piano, int tech=0,
                                   float wf1=2.3f, float wf2=5.7f,
                                   float wp1=0.0f, float wp2=0.0f, float wamp=0.012f){
        // V6.2: Reset confirmation flag BEFORE triggering
        transitionConfirmed.store(false, std::memory_order_seq_cst);
        
        pendingTrans.transTotalFrames = (drwav_uint64)(dur * 44100);
        pendingTrans.p_beats     = beats;  pendingTrans.p_bass_swap = bass;
        pendingTrans.p_echo      = echo;   pendingTrans.p_stutter   = stutter;
        pendingTrans.p_wash      = wash;   pendingTrans.p_bpm       = bpm;
        pendingTrans.p_piano_hold= piano;  pendingTrans.technique_id= tech;
        pendingTrans.w_f1=wf1; pendingTrans.w_f2=wf2;
        pendingTrans.w_p1=wp1; pendingTrans.w_p2=wp2;
        pendingTrans.w_amp=wamp;
        
        // V6.2: Strengthen memory ordering for guaranteed visibility
        std::atomic_thread_fence(std::memory_order_release);
        pendingTrans.pending.store(true, std::memory_order_seq_cst);
    }

    void play(const std::string& n){
        if(n=="A") deckA.isPlaying.store(true,  std::memory_order_relaxed);
        if(n=="B") deckB.isPlaying.store(true,  std::memory_order_relaxed);
    }
    
    void pause(const std::string& n){
        if(n=="A") deckA.isPlaying.store(false, std::memory_order_relaxed);
        if(n=="B") deckB.isPlaying.store(false, std::memory_order_relaxed);
    }

    void seek(const std::string& n, float sec){
        Deck* t = (n=="A") ? &deckA : &deckB;
        drwav_uint64 idx = (drwav_uint64)(sec * t->stride());
        if(t->channels >= 2) idx -= idx % t->channels; 
        
        if(idx < t->totalSampleCount)
            t->currentSampleIndex.store(idx, std::memory_order_relaxed);
    }

    float get_position(const std::string& n){
        Deck* t   = (n=="A") ? &deckA : &deckB;
        drwav_uint64 strideVal = t->stride();
        if(strideVal == 0) return 0.f;
        return (float)t->currentSampleIndex.load(std::memory_order_relaxed)
               / (float)strideVal;
    }

    // V6.2: NEW - Get current volume for multi-signal detection
    float get_volume(const std::string& deck) {
        Deck* d = (deck == "A") ? &deckA : &deckB;
        return d->volume;  // Read current (smoothed) volume
    }
    
    // V6.2: NEW - Manually set volume
    void set_volume(const std::string& deck, float vol) {
        Deck* d = (deck == "A") ? &deckA : &deckB;
        d->tgt_volume = clampf(vol, 0.0f, 1.0f);
    }

    // V6.1: Safe visual buffer read with lock-free handoff
    std::vector<float> get_visual_buffer(const std::string& n){
        Deck* t = (n=="A") ? &deckA : &deckB;
        std::vector<float> ret;
        
        // Try to acquire lock-free snapshot
        if (!t->visualBufferLock.test_and_set(std::memory_order_acquire)) {
            ret = t->visualBufferPythonThread;
            t->visualBufferLock.clear(std::memory_order_release);
        } else {
            // Audio thread is currently writing - return empty buffer rather than blocking
            ret.resize(512, 0.0f);
        }
        return ret;
    }

    // V6.2: Enhanced with confirmation flag - FIXED and exposed
    bool is_transitioning(){
        return inTransition.load(std::memory_order_seq_cst) || 
               pendingTrans.pending.load(std::memory_order_seq_cst) ||
               transitionConfirmed.load(std::memory_order_seq_cst);
    }

    void add_eq_event(int deck_id, float time_sec,
                      float low_mult, float mid_mult, float high_mult){
        Deck* t = (deck_id==0) ? &deckA : &deckB;
        EQEvent ev;
        ev.trigger_sample = (drwav_uint64)(time_sec * t->stride());
        ev.deck_id   = deck_id;
        ev.low_mult  = clampf(low_mult,  0.0f, 2.0f);
        ev.mid_mult  = clampf(mid_mult,  0.0f, 2.0f);
        ev.high_mult = clampf(high_mult, 0.0f, 2.0f);
        ev.fired     = false;
        std::lock_guard<std::mutex> lk(audioMutex);
        eqSchedule.push_back(ev);
        std::sort(eqSchedule.begin(), eqSchedule.end(),
                  [](const EQEvent& a, const EQEvent& b){
                      return a.trigger_sample < b.trigger_sample;
                  });
    }

    void clear_eq_events(){
        pendingClearEQ.store(true, std::memory_order_release);
    }

    // V6.1: Enhanced swap_decks with null safety and filter reset
    void swap_decks(){
        if(inTransition.load(std::memory_order_relaxed) || pendingTrans.pending.load(std::memory_order_relaxed)){
            inTransition.store(false, std::memory_order_relaxed);
            pendingTrans.pending.store(false, std::memory_order_relaxed);
            transitionConfirmed.store(false, std::memory_order_relaxed);
        }

        std::lock_guard<std::mutex> lk(audioMutex);
        
        // V6.1: Safety check - don't swap if either deck is unloaded
        if(!deckA.pSampleData || !deckB.pSampleData){
            std::cerr << "[NeuroMixer] Warning: Cannot swap decks - one or both decks are unloaded\n";
            return;
        }
        
        std::swap(deckA.pSampleData,      deckB.pSampleData);
        std::swap(deckA.totalSampleCount, deckB.totalSampleCount);
        std::swap(deckA.channels,         deckB.channels);

        drwav_uint64 ti = deckA.currentSampleIndex.load(std::memory_order_relaxed);
        deckA.currentSampleIndex.store(deckB.currentSampleIndex.load(std::memory_order_relaxed), std::memory_order_relaxed);
        deckB.currentSampleIndex.store(ti, std::memory_order_relaxed);

        std::swap(deckA.volume,         deckB.volume);
        std::swap(deckA.tgt_volume,     deckB.tgt_volume);
        std::swap(deckA.gate_volume,    deckB.gate_volume);
        std::swap(deckA.transport_gate, deckB.transport_gate);

        bool tp = deckA.isPlaying.load(std::memory_order_relaxed);
        deckA.isPlaying.store(deckB.isPlaying.load(std::memory_order_relaxed), std::memory_order_relaxed);
        deckB.isPlaying.store(tp, std::memory_order_relaxed);

        std::swap(deckA.eq,           deckB.eq);
        std::swap(deckA.fx,           deckB.fx);
        std::swap(deckA.reverb,       deckB.reverb);
        
        bool tl = deckA.isLooping.load(std::memory_order_relaxed);
        deckA.isLooping.store(deckB.isLooping.load(std::memory_order_relaxed), std::memory_order_relaxed);
        deckB.isLooping.store(tl, std::memory_order_relaxed);
        
        std::swap(deckA.loopStartSample,  deckB.loopStartSample);
        std::swap(deckA.loopEndSample,    deckB.loopEndSample);
        std::swap(deckA.targetLoopCount,  deckB.targetLoopCount);
        
        int tcl = deckA.currentLoopCount.load(std::memory_order_relaxed);
        deckA.currentLoopCount.store(deckB.currentLoopCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
        deckB.currentLoopCount.store(tcl, std::memory_order_relaxed);
        
        std::swap(deckA.tail_fade_gain,   deckB.tail_fade_gain);
        std::swap(deckA.in_tail_fade,     deckB.in_tail_fade);
        
        std::swap(deckA.acapellaIntensity, deckB.acapellaIntensity);
        std::swap(deckA.acapellaCurrent,   deckB.acapellaCurrent);
        
        // V6.1: Reset filter state to prevent cross-track artifacts
        deckA.midHPF.reset();
        deckB.midHPF.reset();

        eqSchedule.clear();
        deckA.eq.tgt_ev_low=deckA.eq.tgt_ev_mid=deckA.eq.tgt_ev_high=1.0f;
        deckB.eq.tgt_ev_low=deckB.eq.tgt_ev_mid=deckB.eq.tgt_ev_high=1.0f;
    }

    // V6.2: FIXED - Now properly exposed to Python
    void set_acapella_mode(const std::string& deck, float intensity) {
        Deck* d = (deck == "A") ? &deckA : &deckB;
        float newIntensity = clampf(intensity, 0.0f, 1.0f);
        
        // V6.1: Reset HPF on mode transitions to prevent transient artifacts
        if ((d->acapellaIntensity < 0.001f && newIntensity >= 0.001f) || 
            (d->acapellaIntensity >= 0.001f && newIntensity < 0.001f)) {
            d->midHPF.reset();
        }
        
        d->acapellaIntensity = newIntensity;
    }

    // V6.2: FIXED - Now properly exposed to Python
    float get_acapella_mode(const std::string& deck) {
        return (deck == "A") ? deckA.acapellaIntensity : deckB.acapellaIntensity;
    }
    
    // V6.2: FIXED - Now properly exposed to Python
    uint64_t get_lock_failure_count() const {
        return lockFailureCount.load(std::memory_order_relaxed);
    }
    
    // V6.2: FIXED - Now properly exposed to Python
    void reset_lock_failure_count() {
        lockFailureCount.store(0, std::memory_order_relaxed);
    }
    
    // V6.2: NEW - Diagnostic heartbeat check - FIXED and exposed
    uint64_t get_callback_heartbeat() const {
        return callbackHeartbeat.load(std::memory_order_relaxed);
    }
};

// ── V6.2: COMPLETE Python Bindings (ALL methods properly exposed) ────────────
PYBIND11_MODULE(neuro_core, m){
    m.doc() = "Neuro-DJ Audio Engine v6.2 - Professional DJ transition engine with real-time DSP";
    
    py::class_<NeuroMixer>(m, "NeuroMixer")
        .def(py::init<>())
        .def("load_deck",   &NeuroMixer::load_deck,
             py::arg("name"), py::arg("filepath"),
             "Load audio file into deck A or B (44.1kHz WAV only)")
        .def("play",        &NeuroMixer::play,
             py::arg("deck"),
             "Start playback on specified deck (click-free)")
        .def("pause",       &NeuroMixer::pause,
             py::arg("deck"),
             "Pause playback on specified deck (click-free)")
        .def("seek",        &NeuroMixer::seek,
             py::arg("deck"), py::arg("time_sec"),
             "Seek to position in seconds")
        .def("get_position",&NeuroMixer::get_position,
             py::arg("deck"),
             "Get current playback position in seconds")
        .def("get_volume",  &NeuroMixer::get_volume,
             py::arg("deck"),
             "Get current volume level (0.0-1.0)")
        .def("set_volume",  &NeuroMixer::set_volume,
             py::arg("deck"), py::arg("volume"),
             "Set target volume level (0.0-1.0)")
        .def("swap_decks",  &NeuroMixer::swap_decks,
             "Swap all state between deck A and B (for re-queueing)")
        .def("is_transitioning",  &NeuroMixer::is_transitioning,
             "Check if transition is currently active (multi-signal detection)")
        .def("get_visual_buffer", &NeuroMixer::get_visual_buffer,
             py::arg("deck"),
             "Get lock-free waveform buffer for visualization (512 samples)")
        .def("trigger_hybrid_transition", &NeuroMixer::trigger_hybrid_transition,
             py::arg("duration"), py::arg("beats"), py::arg("bass_swap_point"),
             py::arg("echo_intensity"), py::arg("stutter_intensity"), py::arg("wash_intensity"),
             py::arg("bpm"), py::arg("piano_hold"), py::arg("technique")=0,
             py::arg("wobble_f1")=2.3f, py::arg("wobble_f2")=5.7f,
             py::arg("wobble_p1")=0.0f, py::arg("wobble_p2")=0.0f, py::arg("wobble_amp")=0.012f,
             "Execute AI-designed transition with specified technique (0-8)")
        .def("set_loop_region",  &NeuroMixer::set_loop_region,
             py::arg("deck_id"), py::arg("start_sec"), py::arg("end_sec"), py::arg("loop_count"),
             "Set loop region with automatic count (0 = infinite)")
        .def("add_eq_event",     &NeuroMixer::add_eq_event,
             py::arg("deck_id"), py::arg("time_sec"),
             py::arg("low_mult"), py::arg("mid_mult"), py::arg("high_mult"),
             "Schedule timed EQ change (for drops/builds)")
        .def("clear_eq_events",  &NeuroMixer::clear_eq_events,
             "Clear all scheduled EQ events")
        .def("set_acapella_mode", &NeuroMixer::set_acapella_mode, 
             py::arg("deck"), py::arg("intensity"),
             "Enable mid/side vocal isolation (0.0 = off, 1.0 = full)")
        .def("get_acapella_mode", &NeuroMixer::get_acapella_mode, 
             py::arg("deck"),
             "Get current acapella extraction intensity")
        .def("get_lock_failure_count", &NeuroMixer::get_lock_failure_count,
             "Get number of audio callback lock failures (diagnostic)")
        .def("reset_lock_failure_count", &NeuroMixer::reset_lock_failure_count,
             "Reset lock failure counter")
        .def("get_callback_heartbeat", &NeuroMixer::get_callback_heartbeat,
             "Get audio callback heartbeat counter (diagnostic)");
}