"""Build a concise paper source without changing equations, tables or references."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'Docs/Combined Paper'
TITLE = 'Relative-Depth-Guided Residual Learning for Underwater Image Enhancement'
original = (OUT / (TITLE + '.md')).read_text()
equations = re.findall(r'(?s)\\\[.*?\\\]', original)
tables = re.findall(r'(?m)^\|.*(?:\n\|.*)*', original)
references = original.split('## References\n\n')[1].split('## Appendix A.')[0].strip()
header = original.split('## Abstract')[0].replace('Jayanti Dansena', 'Jayanti Dansana')
text = r'''## Abstract

A local, native-resolution underwater enhancement prototype combines a residual U-Net, full-input colour statistics and frozen monocular relative depth. Scalar and selective fusion use raw L1 and bounded SSIM objectives. Across 40 repeatedly inspected diagnostic images, both depth architectures improve PSNR but reduce SSIM against matched RGB checkpoints. Selective epoch20 obtains 20.993416 dB / 0.878263 versus RGB's 20.907860 dB / 0.881212. Interventions show useful and harmful depth dependence. The experimental MVP supports live depth, cancellation and native-size sRGB PNG export. On an Intel i7-1255U, 30 operations per diagnostic input after three warmups yield p95 worker-pipeline times of 3.360, 6.314 and 8.361 s at 276×182, 1280×803 and 1800×1350. Browser/upload transport is excluded. Functional profiling does not override failed quality gates or replace reserved final evaluation.

**Keywords:** underwater image enhancement; relative depth; residual learning; U-Net; structural similarity; image fidelity.

## 1. Introduction and Related Work

Wavelength-dependent attenuation and backscatter degrade underwater colour and contrast; atmospheric dehazing assumptions do not fully describe underwater formation [1]. Enhancement must also preserve objects, boundaries and spatial relationships without clipping, halos or amplified noise. This study tests relative depth as an auxiliary scene representation through a controlled RGB comparison. Estimated depth is neither measured distance nor water-property calibration.

UIEB supplies real underwater images, curated references and Water-Net [2]. FUnIE-GAN [3] and Fabbri et al. [6] explore adversarial enhancement, with the latter also assessing diver tracking. Colour-balance fusion [4] and transmission-guided RGB/HSV/Lab embedding in Ucolor [5] address colour and spatial degradation. Here, reconstruction uses pixel/structural objectives without adversarial refinement or downstream tracking evaluation.

U-Net [7] motivates multiscale encoder-decoder reconstruction; residual learning [8] motivates corrections to the input. U-Shape Transformer and LSUI [9] provide an alternative using global spatial modeling and channel-wise fusion. The implemented convolutional model instead combines residual RGB correction, input-statistics colour adjustment and selective depth fusion.

Depth Anything V2 Small [10] supplies frozen relative depth; the earlier Depth Anything [11] provides pretrained-depth context. Underwater transfer requires evaluation. Sea-Thru uses RGBD and physical modeling [12], whereas this system infers uncalibrated depth from RGB. Prior depth guidance includes a depth-conditioned GAN [13], WaterGAN's RGB-depth-based synthesis [14] and DAUT's depth-aware Transformer [15]. Wavelength compensation/dehazing [16] and single-image depth/transmission estimation [17] offer physical precedents. Depth guidance itself is therefore not claimed as novel. Contributions concern the implemented residual formulation, fidelity/artifact protocol and local application; no matched external-method benchmark is available.

## 2. Methodology

### 2.1 Residual and colour formulation

For oriented degraded RGB \(I\in[0,1]^{H\times W\times3}\), reference \(J\), and normalized frozen depth \(D=\mathcal{N}(\mathcal{R}(D_\psi(I)))\), the V3 backbone predicts:

@EQ0@

The six statistics \(s\) are full-native-input RGB means and population standard deviations. A 6→32→6 MLP uses tanh heads to bound gain changes to +/-0.5 and offsets to +/-0.25. Training crops retain parent-input statistics before padding/cropping; inference computes complete-input moments. Shared flips/right-angle rotations leave them invariant. RGB controls retain the colour branch and omit depth. Only the local residual is bounded by +/-0.5; combined raw corrections can reach +/-1.25. Metrics/export use bounded predictions, while L1 supervises raw predictions. Curated appearance targets do not establish physical radiance or recovery of absent detail.

### 2.2 Preparation and fusion

Shared loading/inference applies EXIF orientation once, converts valid embedded profiles to sRGB and interprets untagged RGB as sRGB. Unsupported modes, multiframe inputs, malformed profiles and resource-limit violations produce explicit errors. Frozen-estimator output is bilinearly restored to native dimensions with align_corners=False, then normalized:

@EQ1@

Depth is relative, not metric or calibrated confidence. Cached maps bind source-image, estimator and preprocessing identities. The four RGB scales use widths 16/32/64/128; depth widths are 4/8/16/32. Decoder upsampling and matching encoder skips retain spatial features. Spatial normalization is excluded to avoid dependence on crop-versus-full-image statistics. At scale l, projected depth enters through a scalar gate and, for selective fusion, a per-pixel/channel selector:

@EQ2@

Both selector convolutions are 1×1 with eight hidden channels. Scalar gates initialize to zero; zero selector-final weights give \(S_l=0.5\), while scalar-only fusion uses \(S_l=1\). Identity-centred initialization also requires zero local/colour output layers. Selectors affect amplitude and optimization as well as adaptivity; gates are not confidence estimates. Bottom/right replication padding satisfies the factor-of-eight backbone and is removed afterward, preserving oriented dimensions without scene resizing/cropping.

### 2.3 Reconstruction objective

@EQ3@

Weights are \(\lambda_1=1.0\), \(\lambda_s=0.2\). Raw L1 preserves a recovery gradient for pre-clamp values outside [0,1]; bounded SSIM measures local structural agreement [18]. Pixel/structural loss research [19] motivates their joint examination but does not validate these weights or clipping policy. Perceptual and adversarial weights are zero; no refinement result is reported.

## 3. Dataset and Training

UIEB contains 890 paired and 60 challenging images [2]; references are curated targets, not calibrated undegraded scenes. Auditing retains 883 pairs and 51 supported challenging images. Seven duplicate paired inputs and nine challenging originals (unsupported multiframe/source-overlap issues) are excluded from partition membership; original bytes/provenance remain preserved.

Table I. Audited membership. RGB selection was used; final confirmation remains reserved.

@TABLE0@

Conservative visual/source-proxy groups define membership. Missing capture-site, camera and dive identities prevent claims of acquisition-domain/geographic independence. The 51-image subset is not the complete 60-image challenging benchmark.

Training uses synchronized 256×256 RGB/reference/cached-depth crops, rotations and horizontal/vertical flips; cached-depth runs exclude photometric augmentation. Settings are batch 2, AdamW, learning rate \(2\times10^{-4}\), weight decay \(10^{-4}\), seed 20261101 and 20 epochs, with candidates at 5/10/20. Runs share membership, V3 RGB backbone, schedule and reconstruction policy. Generator parameters are RGB 487,753; scalar depth 511,881; selective depth 517,913, excluding frozen depth. Two depth architectures/six candidates versus one RGB run/three candidates create unequal search opportunity. Earlier pilots and repeated diagnostic inspection add development history; inference-time depth removal is not a separately trained control.

## 4. Evaluation and Application

### 4.1 Fidelity and selection protocol

Primary metrics use native oriented float RGB, removed padding, data range 1, no resizing/cropping/tiling and equal image weighting. Channel-wise SSIM uses an 11×11 Gaussian window, sigma 1.5, \(K_1=0.01\), \(K_2=0.03\), population moments and valid-window averaging [18]. Images below the window size on either axis are rejected; implementation/protocol identities accompany results. Secondary 256×256 scores remain separate. External comparisons require matched partitions, preprocessing, metrics and checkpoint conditions. UCIQE (chroma dispersion, luminance contrast, saturation) [20] and UIQM (colourfulness, sharpness, contrast) [21] are relevant but unmeasured here.

Artifact review examines scene colour, boundaries, highlights, texture and noise. Major defects introduce/worsen false structure, recognizable-content loss, strong halos or severe noise; lesser visible correction defects are minor. Pre-existing degradation is distinguished from introduced defects. Model-masked review keeps originals visible and records reviewer/method/disagreements; AI review is identified. Unresolved disagreements retain higher severity. Eligibility requires zero major and at most 10% minor cases. Eligible candidates maximize native selection PSNR; ties within 0.01 dB use SSIM, then earlier step. Metrics cannot override artifact rejection.

RGB, scalar-depth and selective-depth runs are completed; no trained concatenation-only result exists. Depth omission/perturbation tests sensitivity. Planned final confirmation freezes candidates/settings before unseen paired/challenging evaluation, using scene-cluster resampling of paired differences with shared model resamples. It remains unperformed: no confidence intervals/significance are claimed. Such intervals would address image sampling, not arbitrary seeds/domains.

### 4.2 Local workflow

Gradio accepts single-frame RGB JPEG/PNG up to 20 MiB and 2.5 million decoded pixels [E4]. Upload, validation, processing, labeled comparison and export preserve native output dimensions. Lossless RGB PNG embeds sRGB and strips source EXIF/GPS/thumbnails. Isolated workers and operation identifiers enable cancellation, stale-result rejection and retry; new/invalid uploads clear output/download state. Checkpoint incompatibility, depth failure, invalid input, memory exhaustion and export errors are explicit, with no silent RGB fallback. Local temporary-file cleanup and bundles identifying weights, architecture, preprocessing and checksums support reproducibility.

## 5. Results and Discussion

### 5.1 Diagnostic fidelity and depth dependence

Each run completed 20 epochs/7,140 updates and three checkpoints [E1]. Tables II-III use all 40 repeatedly inspected diagnostic images and the fixed native float protocol, not held-out benchmark data.

Table II. Native diagnostic PSNR (dB).

@TABLE1@

Table III. Native diagnostic SSIM.

@TABLE2@

Both depth variants raise PSNR and lower SSIM at every matched epoch. Selective epoch20 exceeds scalar depth by 0.057010 dB / 0.000014 SSIM but is 0.002950 below RGB SSIM. Continuation required a candidate to match/exceed both RGB means without regressing either scalar-depth mean; none passed [E3]. Formal depth selection and final paired/challenging confirmation remain unevaluated.

Selective sensitivity tests on fish14/wreck380/equipment729 use native, zero, horizontally flipped depth and fusion removal. Removing fusion changes fish PSNR 26.284351→22.127201 and wreck 18.315401→17.493034, but improves equipment PSNR 20.422432→21.338726 and SSIM 0.909081→0.921352. These out-of-distribution interventions show mixed dependence, not trained-control results. A separate scalar equipment test found exact live/cache depth equality; no stale-cache mismatch was demonstrated.

RGB's separate 40-image selection review failed [E2]: combined minor counts were 19/34/31 at epochs 5/10/20, with zero reported major cases. Two procedurally isolated AI passes disagreed at epoch20 (3 versus 31 minor); higher severity was retained. Human consensus/statistical reviewer independence are not claimed. The deployed experimental model retains this rejection; depth checkpoints lack formal artifact selection.

### 5.2 Functional MVP and CPU profiling

Selective epoch20 was frozen for integration as an experimental MVP without a passed quality gate [E4]. Each operation loads fresh depth/generator models and performs live depth plus native PNG export. Successive API uploads at 1280×803 and 1800×1350 produced changed sRGB RGB outputs without source EXIF; observed depth-stage cancellation terminated the worker, retry succeeded and corrupt upload cleared results. The recorded fixture run passed 63 tests, including depth-failure propagation without RGB fallback. Provisional worker observations were 8.049 s / 1.539 GB (14_img_) and 8.339 s / 2.540 GB (734_img_); cancel RPC took 0.062 s. These small-sample measurements exclude upload/browser display and parent/UI RSS and differ in timing scope from Table IV.

Repeated profiling used Intel Core i7-1255U, Linux 7.2.13-200.fc44.x86_64, glibc 2.43 and Python 3.14.8 [E5]. Three excluded warmups and 30 measurements per input yield 90 measured/9 warmup operations. Every operation starts a fresh worker/models with live depth; warmups warm OS caches, not resident models. All 99 identities/order/dimensions/hashes/live-depth flags were checked and JSONL aggregates independently recomputed without inference.

Table IV. Worker-pipeline CPU profile: 30 measurements and 3 excluded warmups per image; decimal GB.

@TABLE3@

Timing begins before validation/session setup and ends at exported worker output, including process/model startup, depth, enhancement and export. Browser/network/upload transport and subsequent reopen verification are excluded; RSS excludes parent/UI. p95 uses NumPy default linear interpolation. These three diagnostic-image distributions do not establish real-time speed, resolution-class worst cases, latency confidence intervals, memory safety margins or the full device/resource contract. Selection/final/challenging images were not profiled. Later browser-export verification found fish pixels identical to the packaged output [E4]; API checks/parity do not establish full manual usability approval or reverse quality rejection.

### 5.3 Qualitative evidence

Figures 2-3 show raw input, separately trained RGB20, selective-depth MVP20 and curated reference for fish14, diver/coral887, low-resolution284, wreck380, equipment729 and reef/hand851. These repeatedly inspected development examples are illustrative, not independent/blinded evaluation [E1], [E3]. Wreck retains green haze/reference-colour mismatch; equipment shows harmful depth dependence; reef/hand has pale/pink highlights. Fish correction/export does not prove physical colour recovery or absent-detail reconstruction. Whole panels and matched native-coordinate crops are display-scaled; provenance retains coordinates, hashes and native-float scores.

## 6. Conclusion and Reproducibility

The system implements native-resolution residual/colour enhancement, selective relative-depth fusion and cancellable local export. Diagnostic PSNR gains coexist with SSIM regressions and mixed depth dependence. A functional profiled prototype is supported; consistent depth benefit and research-release acceptance are not. RGB artifact selection and depth continuation failed; formal depth selection/final confirmation remain unavailable. Broader data, multiple seeds, further artifact review and frozen-model held-out evaluation remain future requirements.

Metrics/checkpoint/source identities are in Code/artifacts/phase3/mvp/paper_metrics.csv and paper_evidence.json; profiles and verified aggregates are in profile/ and profile_verification/ below the same directory. The experimental selective epoch20 SHA-256 is b59fdab375e6be1b80eba42cd96fff8e1af524ef0a52d4e16000fbc96c16e488 [E4], [E5]. No missing benchmark, human rating or interval is imputed. Derived records do not confer unrestricted dataset/weight redistribution; source terms apply. Final results require frozen candidates and held-out protocol completion.

![Implemented selective-depth architecture.](../Integration/assets/figures/architecture.png)

Fig. 1. Residual RGB, full-input colour statistics and selective depth fusion; 517,913 trainable generator parameters exclude frozen depth. Gates are not confidence estimates [E1].

![Diagnostic comparisons: fish, diver/coral and low-resolution scene.](../Integration/assets/figures/qualitative_whole_1.png)

Fig. 2. Input, RGB20, selective MVP20 and reference: fish14, diver/coral887, low-resolution284; native-float scores and inspected diagnostic cases [E1].

![Diagnostic comparisons: wreck, equipment and reef/hand.](../Integration/assets/figures/qualitative_whole_2.png)

Fig. 3. Same columns for wreck380, equipment729 and reef/hand851; retained haze/colour mismatch, harmful equipment depth and pale/pink highlights [E1], [E3].

![Native-coordinate crops for Figure 2.](../Integration/assets/figures/qualitative_native_1.png)

Fig. 4. Matched native-coordinate crops for Fig. 2, scaled for layout. Coordinates/hashes remain in qualitative_provenance.json [E1].

![Native-coordinate crops for Figure 3.](../Integration/assets/figures/qualitative_native_2.png)

Fig. 5. Matched crops for Fig. 3; inspected examples do not establish independent quality or physical-detail recovery [E1].

![Operation lifecycle.](../Integration/assets/figures/lifecycle.png)

Fig. 6. Operation IDs, worker cancellation and stale-output invalidation; observed checks do not establish worst-case cancellation guarantees [E4].

## References

@REFERENCES@

'''
for i, equation in enumerate(equations):
    text = text.replace(f'@EQ{i}@', equation)
for i, table in enumerate(tables):
    text = text.replace(f'@TABLE{i}@', table)
# Restore useful context within the six-page budget rather than leaving a
# lightly occupied final page. Technical tables/equations remain verbatim.
intro = original.split('## 1. Introduction\n\n')[1].split('## 2. Related Work')[0].strip()
start = text.index('Wavelength-dependent attenuation')
end = text.index('\n\nUIEB supplies', start)
text = text[:start] + intro + text[end:]
start = text.index('UIEB supplies')
end = text.index('\n\nU-Net [7] motivates', start)
underwater_context = original.split('### 2.1 Underwater enhancement and benchmark data\n\n')[1].split('### 2.2 Encoder-decoder reconstruction')[0].strip()
underwater_context = '\n\n'.join(underwater_context.split('\n\n')[:2]) + '\n\nColour-balance fusion [4] and transmission-guided RGB/HSV/Lab embedding in Ucolor [5] address colour and spatial degradation. Fabbri et al. examine adversarial restoration and diver tracking [6]; downstream tracking is not evaluated here.'
text = text[:start] + underwater_context + text[end:]
start = text.index('Depth Anything V2 Small')
end = text.index(' Sea-Thru uses', start)
depth_context = original.split('### 2.3 Monocular relative depth\n\n')[1].split('### 2.4 Physics-aware recovery')[0].strip()
text = text[:start] + depth_context + '\n\n' + text[end:].lstrip()
loss_context = original.split('The L1 term uses the prediction before final clipping.')[1].split('\n\nPerceptual and adversarial')[0]
text = text.replace('Raw L1 preserves a recovery gradient for pre-clamp values outside [0,1]; bounded SSIM measures local structural agreement [18].', 'Bounded SSIM measures local structural agreement [18].\n\nThe L1 term uses the prediction before final clipping.' + loss_context)
text = text.replace('@REFERENCES@', references)
# Internal provenance is retained in project records, not the research paper.
text = re.sub(r'\s*\[E\d+\](?:,\s*\[E\d+\])*', '', text)
text = text.replace('## 6. Conclusion and Reproducibility', '## 6. Conclusion')
start = text.index('Metrics/checkpoint/source identities are in Code/')
end = text.index('\n\n![', start)
text = text[:start] + 'Evaluation records retain model identities and preprocessing settings. No missing benchmark, human rating or confidence interval is imputed. Availability of derived records does not confer unrestricted redistribution of the source dataset or pretrained weights; source terms apply. Final confirmation requires frozen candidates and completion of the held-out protocol.' + text[end:]
text = text.replace('Coordinates/hashes remain in qualitative_provenance.json.', 'Matched crop coordinates and image identities are retained in evaluation records.')
# Place the two system diagrams beside their explanations so indivisible
# figures do not leave large gaps at the end of the text section.
for asset, anchor in [('architecture.png', '### 2.1 Residual and colour formulation'), ('lifecycle.png', '## 5. Results and Discussion')]:
    pattern = r'!\[[^\n]*\]\([^\n]*' + re.escape(asset) + r'\)\n\nFig\.[^\n]*'
    figure = re.search(pattern, text).group(0)
    text = re.sub(pattern + r'\n\n', '', text)
    text = text.replace(anchor, figure + '\n\n' + anchor)
figure_numbers = {'1': '1', '6': '2', '2': '3', '3': '4', '4': '5', '5': '6'}
text = re.sub(r'Fig\. ([1-6])\b', lambda m: 'Fig. ' + figure_numbers[m[1]], text)
text = text.replace('Figures 2-3 show', 'Figures 3-4 show')
text = re.sub(r'Figure ([23])\b', lambda m: 'Figure ' + figure_numbers[m[1]], text)
assert not re.search(r'@(EQ|TABLE|REFERENCES|EVIDENCE)', text)
destination = OUT / (TITLE + ' - 6 pages.md')
destination.write_text(header + text)
print(destination)
