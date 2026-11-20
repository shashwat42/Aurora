"""Merge supplied academic framing and recorded integration evidence; no inference runs."""
from pathlib import Path
import re, sys, csv, json, hashlib
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT.parent/'Docs'
OUT=DOCS/'Combined Paper'
OUT.mkdir(exist_ok=True)
TITLE='Relative-Depth-Guided Residual Learning for Underwater Image Enhancement'
STEM=TITLE
AUTHORS='Jayanti Dansena, Sarthak Dey, Shashwat Singh, Sahil Ballav, Shivam Yadav, Abhi'
AFF='School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar, Odisha, India'
s=(DOCS/'Integration/Evidence_Based_Working_Draft.md').read_text()
s='# '+TITLE+'\n\n'+AUTHORS+'\n\n'+AFF+'\n\n'+s[s.index('## Abstract'):]
abstract=(DOCS/'Integration/02_Ready_to_Integrate.md').read_text().split('## Revised abstract draft\n\n')[1].split('\n\nThe sections')[0]
s=re.sub(r'(?s)(## Abstract\n\n).*?(\*\*Keywords:)',lambda m:m[1]+abstract+'\n\n'+m[2],s,count=1)
s=s.replace('sealed final evaluation partitions','reserved final evaluation partitions')
s=s.replace('## 3. Methodology','''### 2.4 Physics-aware recovery and prior depth conditioning

Sea-Thru uses RGBD observations and a revised underwater formation model to estimate range-dependent colour recovery [7]. Aurora instead uses a relative map inferred from RGB; its output is not a physically calibrated restoration. Depth conditioning is also present in the depth-conditioned GAN of Wu et al. [8], which supplies estimated depth and RGB to an enhancement generator. Accordingly, depth guidance itself is not claimed as novel. This study concerns the implemented residual formulation and its recorded controlled development comparison; no matched external-method benchmark is available.

## 3. Methodology''')
s=s.replace('This preserves identity initially but changes subsequent fusion amplitude and optimization as well as adaptivity.', 'Zero scalar gates initially suppress depth contributions; the selector also changes fusion amplitude and optimization as well as adaptivity. The complete identity-centred initialization additionally depends on the zero-initialized local and colour output layers.')
s=s.replace('The implemented V3 backbone combines','The implemented V3 backbone (Fig. 1) combines')
s=s.replace('Any refinement variant must be evaluated separately under the same fidelity and artifact requirements.', 'No perceptual or adversarial refinement result is reported.')
s=s.replace('Any image', 'Any image')
s=s.replace('Depth selection and final partitions remain unopened.', 'Depth checkpoints have not entered formal selection; final paired and challenging confirmation remain unevaluated.')
s=s.replace('remains0.002950','remains 0.002950').replace('were19/34/31','were 19/34/31').replace('epochs5/10/20','epochs 5/10/20').replace('3 versus31','3 versus 31').replace('and1800','and 1800').replace('All63','All 63')
s=s.replace('All three runs completed 20 epochs/7,140 updates and three checkpoints.', 'All three runs completed 20 epochs/7,140 updates and three checkpoints [9].')
s=s.replace('V3 RGB formal artifact review also failed:', 'The separate 40-image checkpoint-selection partition was evaluated for V3 RGB, and its formal artifact review failed [10]:')
s=s.replace('The depth checkpoints have not undergone a formal selection artifact review.', 'The depth checkpoints have not undergone a formal selection artifact review. Selective continuation required at least one candidate to match or exceed both matched RGB means without regressing either matched scalar-depth mean; none satisfied this rule [11].')
s=s.replace('Final confirmation reserves unseen paired and challenging images until model identities and evaluation settings are frozen. Paired differences in image-quality scores are assessed with scene-cluster resampling, using the same resamples for all compared models. Confidence intervals characterize image-sampling uncertainty for the trained checkpoints; they do not establish robustness across arbitrary training seeds or underwater domains.', 'The planned final-confirmation protocol reserves unseen paired and challenging images until model identities and evaluation settings are frozen. It specifies scene-cluster resampling of paired image-quality differences with shared resamples across models. This analysis has not been performed; no confidence interval or statistical significance claim is reported. Such intervals would address image-sampling uncertainty, rather than arbitrary training seeds or domains.')
s=s.replace('The local interface organizes', 'The implemented local Gradio interface accepts single-frame RGB JPEG/PNG inputs up to 20 MiB and 2.5 million decoded pixels [12]. It organizes')
s=s.replace('All 63 fixture tests passed', 'The recorded fixture-test run passed all 63 tests')
s=s.replace('The Gradio workflow invokes', 'The recorded Gradio workflow [12] invokes')
s=s.replace('A manual browser check of the depth mode and the full original device/resource contract remain outstanding; profiling does not change failed image-quality continuation criteria.', 'A later recorded browser-export check found the downloaded fish output pixel-identical to the packaged result [12]. This functional parity observation is not a complete manual usability approval. The full original device/resource contract remains unverified; profiling does not change failed image-quality continuation criteria.')
s=s.replace('Every operation used a fresh worker', 'The recorded profiling evidence [13] states that every operation used a fresh worker')
# Keep narrow-column result tables legible by splitting the recorded paired metrics.
start=s.index('| Epoch | RGB PSNR / SSIM')
end=s.index('\n\n',start)
rows=list(csv.DictReader((ROOT/'artifacts/paper_integration/tables/diagnostic_metrics.csv').open()))
def metric_table(metric):
    lines=['Table '+('II. Native diagnostic PSNR (dB).' if metric=='psnr' else 'III. Native diagnostic SSIM.'),'','| Epoch | RGB | Scalar depth | Selective depth |','| --- | ---: | ---: | ---: |']
    for epoch in (5,10,20):
        rr=[next(r for r in rows if r['model']==model and int(r['epoch'])==epoch) for model in ('RGB V3','Scalar depth','Selective depth')]
        lines.append('| '+str(epoch)+' | '+' | '.join(f"{float(r[metric]):.6f}" for r in rr)+' |')
    return '\n'.join(lines)
s=s[:start]+metric_table('psnr')+'\n\n'+metric_table('ssim')+s[end:]
s=s.replace('| Partition | Images | Purpose |','Table I. Audited partitions and roles. RGB selection has been used; final confirmation has not.\n\n| Partition | Images | Purpose |')
start=s.index('| Diagnostic input | Dimensions')
end=s.index('\n\n',start)
prof=list(csv.DictReader((ROOT/'artifacts/paper_integration/tables/runtime_profile.csv').open()))
t=['Table IV. Repeated worker-pipeline CPU profile. Each row represents one image, 30 measurements and 3 excluded warmups. GB is decimal.','','| Input (pixels) | Median s | p95 s | Max RSS GB |','| --- | ---: | ---: | ---: |']
for r in prof:t.append(f"| {r['width']} x {r['height']} | {float(r['median_seconds']):.3f} | {float(r['p95_seconds']):.3f} | {int(r['max_worker_rss_bytes'])/1e9:.3f} |")
s=s[:start]+'\n'.join(t)+s[end:]
s=s.replace('## 8. Limitations','''### 7.5 Qualitative evidence

Figures 3-6 reproduce the six packaged diagnostic cases and their fixed-coordinate crops: fish (14), diver/coral (887), low-resolution scene (284), wreck (380), equipment (729), and reef/hand (851). Each panel presents the raw input, separately trained RGB20 output, selective-depth MVP20 output, and curated reference. These examples were already inspected during development; they are illustrations, not an independent qualitative benchmark [9], [11]. Whole-image panels are resized for display, while the crop sheets originate from matched native-pixel coordinates. The packaged provenance records preserve coordinates, input/output hashes and per-image scores.

The integration captions record residual green haze and reference-colour mismatch in the wreck, harmful depth dependence for equipment, and pale/pink highlights in the reef/hand case. The fish output demonstrates an implemented appearance correction and export operation; it does not establish physically verified colour recovery or reconstruction of absent detail. These observations complement the mixed fidelity means and retained artifact rejection.

## 8. Limitations''')
start=s.index('## 9. Conclusion')
end=s.index('## 10.',start)
s=s[:start]+'''## 9. Conclusion

Aurora implements relative-depth-guided residual RGB enhancement with selective multiscale fusion, full-input colour statistics, native-resolution processing and cancellable local export. The completed diagnostic comparisons show higher PSNR but lower SSIM for both depth architectures at every matched checkpoint. At epoch 20, selective depth achieves 20.993416 dB and 0.878263 SSIM, compared with 20.907860 dB and 0.881212 for RGB. Three-case interventions show useful and harmful depth dependence. These findings do not establish consistent depth benefit.

The experimental MVP has recorded functional checks and a repeated CPU operation profile. It remains distinct from a quality-approved research model: RGB artifact selection failed, depth continuation failed, formal depth selection is unavailable, and final confirmation has not been performed. Broader data, multiple training seeds, further artifact assessment and frozen-model held-out evaluation remain requirements for stronger conclusions, rather than completed findings.

'''+s[end:]
# Replace source links with a stable numerical bibliography.
links={'Akkaynak and Treibitz, 2018':1,'Li et al., 2019':2,'Islam et al., 2020':3,'Ronneberger et al., 2015':4,'Yang et al., 2024':5,'Wang et al., 2004':6}
for label,num in links.items():s=re.sub(r'\['+re.escape(label)+r'\]\([^)]+\)',f'[{num}]',s)
s=s[:s.index('## References')]+'''## References

[1] D. Akkaynak and T. Treibitz, “A Revised Underwater Image Formation Model,” Proc. IEEE CVPR, pp. 6723-6732, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html

[2] C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao, “An Underwater Image Enhancement Benchmark Dataset and Beyond,” IEEE Trans. Image Process., vol. 29, pp. 4376-4389, 2020, doi: 10.1109/TIP.2019.2955241. Author manuscript: https://arxiv.org/abs/1901.05495

[3] M. J. Islam, Y. Xia, and J. Sattar, “Fast Underwater Image Enhancement for Improved Visual Perception,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 3227-3234, 2020, doi: 10.1109/LRA.2020.2974710. https://arxiv.org/abs/1903.09766

[4] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, pp. 234-241, 2015. https://arxiv.org/abs/1505.04597

[5] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” Advances in Neural Information Processing Systems, vol. 37, 2024. https://arxiv.org/abs/2406.09414

[6] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600-612, 2004, doi: 10.1109/TIP.2003.819861. https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html

[7] D. Akkaynak and T. Treibitz, “Sea-Thru: A Method for Removing Water From Underwater Images,” Proc. IEEE/CVF CVPR, pp. 1682-1691, 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.html

[8] C. Wu, Z. Liu, L. Qian, and J. Zhou, “Depth-Conditioned GAN for Underwater Image Enhancement,” IEEE Access, vol. 11, pp. 141789-141800, 2023, doi: 10.1109/ACCESS.2023.3339817. https://ieeexplore.ieee.org/document/10345621/

[9] Aurora project, “Paper integration evidence package,” internal development records, 2026. Docs/Integration/Evidence_Based_Working_Draft.md; assets/tables/diagnostic_metrics.csv; assets/evidence/qualitative_provenance.json. Paths under assets resolve to Code/artifacts/paper_integration/.

[10] Aurora project, “V3 selection evaluation and artifact review,” internal selection records, 2026. Docs/Version 3 Selection Review.md; Code/artifacts/paper_integration/tables/rgb_artifact_review.csv.

[11] Aurora project, “Selective-depth diagnostic review and sensitivity records,” internal development records, 2026. Code/artifacts/paper_integration/evidence/selective_review.json and selective_sensitivity.json; Code/artifacts/phase3/selective_depth/config.json.

[12] Aurora project, “Experimental MVP workflow and export parity,” internal functional records, 2026. Code/artifacts/paper_integration/evidence/workflow.json, browser_output_parity.json and model_decision.json; Code/artifacts/phase3/mvp/tests.log. These records are not peer-reviewed evidence.

[13] Aurora project, “Repeated MVP CPU profiling and aggregate verification,” internal operational records, 2026. Code/artifacts/paper_integration/tables/runtime_profile.csv and evidence/profile_verification.json. Measurement scope: three diagnostic inputs, 90 recorded operations, nine excluded warmups.
'''
s=s.replace('encoder–decoder','encoder-decoder').replace('±','+/-').replace('—','-').replace('–','-')
# Main-paper figures are colocated with the method and qualitative results.
architecture="""![Implemented selective-depth architecture.](../Integration/assets/figures/architecture.png)

Fig. 1. Implemented selective-depth MVP. The frozen relative-depth estimator is excluded from the 517,913 trainable generator parameters. Full-input RGB statistics condition the colour branch. Feature gates are not calibrated confidence estimates [9]."""
s=s.replace('## 3. Methodology\n\n','## 3. Methodology\n\n'+architecture+'\n\n')
s=s.replace('Figures 3-6 reproduce', 'Figures 2 and 3 reproduce')
s=s.replace('and their fixed-coordinate crops:', ':').replace('cases :', 'cases:')
s=s.replace('Whole-image panels are resized for display, while the crop sheets originate from matched native-pixel coordinates.', 'Whole-image panels are resized for display; corresponding fixed-coordinate native-pixel crops are provided in Appendix B.')
qualitative="""![Diagnostic whole-scene comparisons, cases 14, 887 and 284.](../Integration/assets/figures/qualitative_whole_1.png)

Fig. 2. Raw input, RGB20 control, selective-depth MVP20 and curated reference for fish, diver/coral and low-resolution cases. Panels are resized for display; scores refer to native float predictions. All three cases are repeatedly inspected diagnostic data [9].

![Diagnostic whole-scene comparisons, cases 380, 729 and 851.](../Integration/assets/figures/qualitative_whole_2.png)

Fig. 3. Wreck, equipment and reef/hand diagnostic cases in the same column order as Fig. 2. Integration captions identify residual haze/colour mismatch, harmful depth dependence for equipment, and pale/pink highlights [9], [11]. These limitations accompany the numerical results."""
s=s.replace('## 8. Limitations',qualitative+'\n\n## 8. Limitations')
# Relocate record-navigation details to the supplement without changing evidence.
record_paragraph=s[s.index('Machine-readable cached metrics'):s.index('### 7.4 Repeated operational profiling')].strip()
s=s.replace(record_paragraph+'\n\n','')
profile_paths='Profile records and environment/model identity are preserved in `Code/artifacts/phase3/mvp/profile/`. Verified aggregates and paper-ready CSV are in `Code/artifacts/phase3/mvp/profile_verification/`.'
s=s.replace(profile_paths+' ','')
appendix="""
## Appendix A. Application lifecycle and reproducibility

![Operation lifecycle.](../Integration/assets/figures/lifecycle.png)

Fig. A1. Implemented worker lifecycle and operation identifiers. Cancellation stops the worker and invalidates stale outputs. Recorded workflow checks support the observed operations; they do not establish worst-case cancellation guarantees [12].

### A.1 Recorded evidence locations

@@RECORD_LOCATIONS@@

@@PROFILE_LOCATIONS@@

The experimental selective-depth epoch 20 weights have SHA-256 identity b59fdab375e6be1b80eba42cd96fff8e1af524ef0a52d4e16000fbc96c16e488 [12], [13]. The repeated operational profile and qualitative examples refer to this experimental checkpoint; its identity does not imply quality selection.

## Appendix B. Supplementary native-coordinate comparisons

![Matched native-coordinate crops, cases 14, 887 and 284.](../Integration/assets/figures/qualitative_native_1.png)

Fig. B1. Same-coordinate crops corresponding to main-paper Fig. 2. Crop sheets originate from native pixels and are scaled here for layout. Coordinates and hashes are recorded in qualitative_provenance.json [9]. No verified physical-detail recovery is inferred.

![Matched native-coordinate crops, cases 380, 729 and 851.](../Integration/assets/figures/qualitative_native_2.png)

Fig. B2. Same-coordinate crops corresponding to main-paper Fig. 3. These previously inspected examples are not a blinded independent qualitative benchmark. Display scaling does not alter the dimensions of the underlying native exports [9].
"""
appendix=appendix.replace('@@RECORD_LOCATIONS@@',record_paragraph).replace('@@PROFILE_LOCATIONS@@',profile_paths)
s+=appendix
# The prototype name is not the research framework's name.
s=s.replace("Aurora's", "The study's").replace('Aurora project', 'Project records')
s=s.replace('Aurora is a local', 'The experimental system is a local')
s=s.replace('Aurora', 'The framework')
assert not re.search(r'aurora',s,re.I)
from revise_combined_paper import revise_paper
s = revise_paper(s)
(OUT/(STEM+'.md')).write_text(s)
# Record input provenance and verify already recorded table identities without inference.
used=[Path('/home/shashwatsingh/Downloads/Aurora_Paper_IEEE.pdf')]+list((DOCS/'Integration').glob('*.md'))+list((ROOT/'artifacts/paper_integration/tables').glob('*.csv'))+list((ROOT/'artifacts/paper_integration/evidence').glob('*.json'))+[DOCS/'Version 3 Selection Review.md']
manifest=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in used]
checks=[]
for row in rows:
    p=ROOT/row['summary_path']; ok=hashlib.sha256(p.read_bytes()).hexdigest()==row['summary_sha256'];checks.append({'path':str(p),'sha256_matches':ok});assert ok,p
(ROOT/'artifacts/combined_paper/source_audit.json').write_text(json.dumps({'inputs':manifest,'metric_source_checks':checks,'training_or_inference_launched':False},indent=2))
import subprocess
subprocess.run([sys.executable, str(ROOT / 'scripts/render_combined_paper.py')], check=True)
