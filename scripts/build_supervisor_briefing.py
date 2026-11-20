"""Author a supervisor explanation from local implementation and frozen evidence."""
import csv, hashlib, json, re, sys
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT.parent/'Docs/Integration'
ASSETS=ROOT/'artifacts/paper_integration'
sys.path.append('/home/shashwatsingh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
pdfmetrics.registerFont(TTFont('BriefSans','/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('BriefSans-Bold','/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf'))
pdfmetrics.registerFontFamily('BriefSans',normal='BriefSans',bold='BriefSans-Bold',italic='BriefSans',boldItalic='BriefSans-Bold')
ST={
'body':ParagraphStyle('body',fontName='BriefSans',fontSize=10,leading=14,spaceAfter=9),
'caption':ParagraphStyle('caption',fontName='BriefSans',fontSize=8.5,leading=11.5,textColor=colors.HexColor('#475569'),spaceAfter=9),
'head':ParagraphStyle('head',fontName='BriefSans-Bold',fontSize=19,leading=23,spaceAfter=15,textColor=colors.HexColor('#0f172a')),
'sub':ParagraphStyle('sub',fontName='BriefSans-Bold',fontSize=12,leading=16,spaceAfter=8,textColor=colors.HexColor('#0f172a')),
'cell':ParagraphStyle('cell',fontName='BriefSans',fontSize=9,leading=12,spaceAfter=0),
}
story=[]; md=[]
def para(s,kind='body'):
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escape(s))
    story.append(Paragraph(text,ST[kind]));md.append(s+'\n')
def page(title):
    if story:story.append(PageBreak())
    story.append(Paragraph(escape(title),ST['head']));md.append('## '+title+'\n')
def sub(s):
    story.append(Paragraph(escape(s),ST['sub']));md.append('### '+s+'\n')
def table(rows,widths):
    cells=[[Paragraph(escape(str(c)),ST['cell']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t,Spacer(1,10)])
    md.extend(['| '+' | '.join(map(str,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows[1:]]+[''])
def picture(name,width,caption):
    p=ASSETS/'figures'/name
    with PILImage.open(p) as im:h=width*im.height/im.width
    story.append(Image(str(p),width=width,height=h,hAlign='CENTER'))
    md.append(f'![{caption}](assets/figures/{name})\n');para(caption,'caption')
def evidence(s):para('Evidence: '+s,'caption')

# Assert factual anchors before authoring.
config=json.loads((ROOT/'artifacts/phase3/selective_depth/config.json').read_text())
decision=json.loads((ROOT/'artifacts/phase3/mvp/model_decision.json').read_text())
workflow=json.loads((ASSETS/'evidence/workflow.json').read_text())
profile=json.loads((ROOT/'artifacts/phase3/mvp/profile/summary.json').read_text())
assert config['epochs']==20 and config['batch_size']==2 and config['seed']==20261101
assert decision['weights_sha256']=='b59fdab375e6be1b80eba42cd96fff8e1af524ef0a52d4e16000fbc96c16e488'
assert not decision['formally_selected'] and workflow['status']=='passed'
metrics=list(csv.DictReader((ASSETS/'tables/diagnostic_metrics.csv').open()))

page('Aurora: supervisor briefing')
para('A plain-language explanation of the research idea, implemented model, experimental evidence and working MVP. Prepared 2 November 2026.')
sub('The project in one sentence')
para('Aurora is a local underwater image-enhancement application that uses a trained residual U-Net, an input-based colour-correction branch and a frozen relative-depth estimator to produce an enhanced image at the original oriented resolution.')
sub('A 60-second explanation')
para('We built an image-to-image system: it receives one underwater photograph and predicts a corrected RGB photograph. The central research question is whether estimated scene depth adds useful information beyond the RGB image alone. We trained an RGB control, a scalar depth-fusion model and a selective depth-fusion model. The working application uses the final selective-depth checkpoint and supports validation, live depth estimation, enhancement, comparison, cancellation and PNG download.')
para('We have demonstrated the application workflow and measured its CPU operation. On the 40 development images, depth improves PSNR but reduces SSIM relative to the matched RGB control. Therefore, our result is a functional experimental prototype with mixed quality evidence, rather than a demonstrated general improvement from depth guidance.')
table([['Question','Answer to remember'],['Model type','Deep learning: a CNN residual U-Net with selective multiscale depth fusion and a small colour MLP.'],['What was trained?','The enhancement generator, including its RGB/depth encoders, fusion, decoder and colour branch.'],['What stayed frozen?','Pretrained Depth Anything V2 Small, which supplies relative depth.'],['Current quality figures','Selective epoch 20: PSNR 20.9934 dB; SSIM 0.878263, on 40 diagnostic images.'],['Current product status','Working local experimental MVP; formal depth quality selection and final confirmation remain incomplete.']],[125,385])
evidence('Active model decision; diagnostic metrics; recorded live workflow and verified operational profile. Detailed record map appears at the end.')

page('1. How the earlier paper relates to the built system')
para('Your team paper, Aurora: Relative-Depth-Guided Residual Learning for Underwater Image Enhancement, provides the motivation and the structure of the study: paired enhancement, frozen relative depth, residual reconstruction, controlled comparison, artifact assessment and a local interface. Its project-status section describes an earlier Phase 1 state. Use the current records when explaining completed work.')
sub('Research question and contribution')
para('The question is whether a depth representation inferred from a degraded photograph helps a compact enhancer match a curated reference while preserving image fidelity. Depth is derived from the same photograph: it is an auxiliary representation, not an independent measurement. The contribution is the compact integration and its controlled evaluation, together with an implemented local workflow. Depth guidance itself is not a new invention claimed by this project.')
table([['Paper element','What the implementation now contains'],['Residual image reconstruction','Four-scale residual U-Net plus a separate full-image RGB-statistics colour branch. The original local-residual-only equation is incomplete for the current model.'],['Fusion mechanism','Trained scalar-gated and selective additive fusion. The generic concatenation/projection formula in the earlier paper is not the implemented fusion used for the reported results.'],['Training and evaluation','Three completed 20-epoch runs; diagnostic results at epochs 5, 10 and 20; sensitivity interventions and retained quality decisions.'],['Local application','Live depth, interruptible worker inference, native-size comparison/export and measured CPU profile.'],['Final confirmation','Still reserved. Current diagnostic measurements are repeatedly inspected development evidence.']],[125,385])
para('When explaining progress, distinguish implementation completion from research acceptance. Running the model and producing a file establish functionality; passing the declared quality criteria requires separate evidence. The project has completed depth-model experiments within Phase 3, but that does not mean every original Phase 3 gate has passed.')
evidence('Team paper Sections III-VIII; src/aurora/models/; artifacts/phase3/mvp/model_decision.json; Docs/PHASE3_STATUS.md.')

page('2. Data: what the model learns from')
para('This is supervised paired image-to-image learning. Each training example has a degraded underwater RGB image and a corresponding curated enhanced reference. The model updates its weights so that its output better matches that reference. At inference, only the degraded photograph is required; the reference is used for training and paired evaluation, not supplied to the application.')
para('The UIEB references are curated enhancement targets. Agreement with them measures agreement with the dataset target appearance, not verified restoration of the original physical light or water conditions.')
table([['Partition','Count','Role and current use'],['Training','713','Used to optimize model weights.'],['Diagnostic development','40','Used repeatedly for debugging, model comparisons, sensitivity checks and the reported quality means.'],['Checkpoint selection','40','Reserved for formal screening/ranking. Earlier RGB V3 candidates were evaluated here; the depth candidates have not entered formal selection.'],['Final paired confirmation','90','Reserved for frozen-model final paired evaluation.'],['Challenging confirmation subset','51','Reserved for non-reference assessment; separate from the paired partitions.']],[130,45,335])
para('The audited collection has 883 paired members after excluding seven repeated inputs from partition membership. The supported challenging subset contains 51 of the published 60 images, after exclusions for compatibility/source-overlap issues. Original files and provenance are retained.')
sub('Why use several partitions?')
para('Training adjusts parameters. Diagnostic inspection influences development decisions. Selection chooses an eligible checkpoint under the declared review rule. Final confirmation estimates performance after choices are frozen. Reusing diagnostic images during development makes them informative for debugging, but unsuitable as untouched final evidence.')
para('Visual/source-proxy grouping reduces obvious related-image leakage. Capture-site, camera and dive identifiers are missing, so the split cannot establish independence across acquisition environments. There are no reported final-set results in this briefing.')
evidence('data/metadata/manifest.v1.json; Docs/Data Review.md; Docs/Research Paper Draft.md Section 4; artifacts/phase2/evaluation_v3/.')

page('3. Architecture: follow one image through the model')
picture('architecture.png',500,'Implemented selective-depth architecture. The frozen estimator is separate from the 517,913-parameter trainable generator.')
para('The RGB encoder extracts features at four scales, with 16, 32, 64 and 128 channels. Small-scale features preserve local appearance; coarser features supply wider image context. Convolution applies learned filters over neighbourhoods rather than treating each pixel as an unrelated record.')
para('A depth encoder extracts aligned features with 4, 8, 16 and 32 channels. Projection layers match their channel sizes to RGB features, then selective fusion adds a controlled depth contribution at each scale. The decoder upsamples and combines matching encoder features through skip connections to predict a local RGB correction.')
para('A separate colour branch receives six statistics from the entire input: RGB means and population standard deviations. A 6-to-32-to-6 MLP predicts per-channel gain changes and offsets. It supplies whole-scene colour context alongside the local U-Net correction. The RGB control retains this same colour branch.')
evidence('src/aurora/models/rgb.py, scene_rgb.py, scene_depth.py and selective_depth.py.')

page('4. What residual learning and selective fusion mean')
sub('Residual output: learn a correction to the input')
para('Instead of predicting the complete output independently, the network starts from the input image and adds corrections. In compact notation:')
para('J_pre = I + 0.5 tanh(G(I,D)) + delta_gain(s) * I + offset(s)\nJ_out = clip(J_pre, 0, 1)')
para('I is the RGB input, D the aligned relative-depth map, s the six full-input statistics, and G the local network. The multiplication by delta_gain is per colour channel. Gain changes are bounded by +/-0.5 and offsets by +/-0.25. Only the local residual has the +/-0.5 bound; the total correction is broader. Final clipping makes the output valid RGB.')
sub('Fusion: learn where and how much depth to use')
para('At each scale, projected depth features Z and RGB features F determine a selector S. Two 1 x 1 convolutions with an eight-channel hidden layer and a sigmoid produce S for each location and channel. The fused features are:')
para('F_fused = F + tanh(a) * S * Z')
para('The scalar a is learned separately for each of four scales. The scalar-only depth model uses the same rule with S = 1. The selective version can reduce individual depth-feature contributions depending on local RGB/depth features. Its selectors are learned feature weights, not calibrated probabilities that a depth estimate is correct.')
para('Gates initialize to zero; the selector output initially equals 0.5 because its final layer starts at zero. The local output head and colour output layer are zero-initialized too. This gives an identity-centred starting mapping. Training subsequently changes it; residual learning does not guarantee artifact-free images.')
sub('Depth is relative and frozen')
para('Depth Anything V2 Small is loaded from local pinned files with gradients disabled. Its internally resized prediction is restored to input dimensions by bilinear interpolation. Per-image min-max normalization maps it to [0,1]; a range of at most 1e-6 becomes a zero map. Values do not represent metres. There is no underwater depth-ground-truth supervision in these runs.')
evidence('src/aurora/models/selective_depth.py; scene_rgb.py; scene_depth.py; src/aurora/depth.py.')

page('5. Training: how the learned correction was obtained')
para('Three models were trained from scratch under matched per-run reconstruction settings: RGB V3, scalar-depth and selective-depth. The pretrained depth estimator was not retrained. During training/evaluation, verified cached maps avoid repeatedly running that fixed estimator; the live app deliberately predicts depth for every operation.')
table([['Setting','Recorded value'],['Training membership','713 images per epoch'],['Input crops / batch','256 x 256 / batch size 2'],['Schedule','20 epochs; 7,140 optimizer updates; saved candidates at epochs 5, 10 and 20'],['Optimizer','AdamW, constant learning rate 0.0002; weight decay 0.0001'],['Primary seed','20261101'],['Geometric augmentation','Shared horizontal/vertical flips and right-angle rotations for input, reference and depth'],['Colour statistics','Computed from the full parent input before crop/padding; no target statistics'],['Loss','Mean absolute error on raw prediction + 0.2 x (1 - SSIM on clipped prediction)'],['Trainable parameters','RGB: 487,753; scalar depth: 511,881; selective depth: 517,913; frozen estimator excluded'],['Selective run checks','Finite recorded losses/gradients; exact checkpoint reload; 4,263.54 seconds recorded training time']],[130,380])
para('The L1 component penalizes average absolute pixel error. SSIM adds a local structural agreement objective. L1 uses the pre-clipped prediction because a value outside [0,1] would otherwise lose its direct corrective gradient through hard clipping. SSIM uses bounded RGB. The trained objective contains reconstruction losses; perceptual and adversarial weights are zero.')
para('Weights are the numerical coefficients learned by the network. AdamW updates them using gradients from the loss; the learning rate controls the update scale and weight decay regularizes the weights. A seed fixes the requested random initialization and sampling sequence for reproducibility. These settings describe the experiment, not a guarantee of identical behaviour on every platform.')
para('An epoch is one pass through the training membership; a step is one optimizer update for a batch. Shared geometric transforms preserve alignment between inputs and targets. Photometric augmentation is omitted because changing image appearance without recomputing cached depth would change that relationship.')
para('The runs match per-run settings, not total search opportunity: two depth architectures/six checkpoints were explored against one V3 RGB run/three checkpoints, with earlier RGB/pilot work also in the history. One primary seed does not demonstrate robustness across random initializations.')
evidence('artifacts/phase3/selective_depth/config.json and completion records; src/aurora/training/run.py; Docs/Selective Depth Fusion Results.md.')

page('6. Quality metrics: how to read the results')
sub('PSNR: pixel fidelity to the reference')
para('For bounded RGB values, per-image MSE is the mean squared difference between output and reference across pixels and channels. PSNR = -10 log10(MSE), because the data range is 1. Higher PSNR means lower squared pixel error. It is expressed in decibels, not a percentage of correctness.')
sub('SSIM: local structural agreement')
para('SSIM compares local brightness, contrast and structure with the reference. Higher is better; 1 represents identical images under the implementation. Aurora uses an 11 x 11 Gaussian window, sigma 1.5, K1 = 0.01 and K2 = 0.03, population moments and valid-window averaging over RGB channels. A value of 0.878 is not an 87.8% accuracy score.')
para('Both metrics use aligned native-resolution float RGB before PNG quantization, with output padding removed. Each image has equal weight in the reported mean. No resize/crop is applied for the primary metric. These rules matter because changing resolution, colour handling or SSIM details changes the comparison.')
rows=[['Epoch','RGB PSNR / SSIM','Scalar depth','Selective depth']]
for epoch in [5,10,20]:
 rows.append([str(epoch)]+[f"{float(r['psnr']):.6f} / {float(r['ssim']):.6f}" for r in metrics if int(r['epoch'])==epoch])
table(rows,[40,157,157,156])
para('At epoch 20, selective depth changes the RGB mean by +0.085556 dB PSNR and -0.002950 SSIM. These are small mixed differences, not uniform quality improvement. Epoch 10 has the highest selective-depth diagnostic PSNR, while epoch 20 has its highest diagnostic SSIM. Neither is formally quality-selected.')
para('Selective continuation required at least one checkpoint to meet or exceed both matched RGB means without regressing either matched scalar-depth mean. None did. This is a failed development continuation rule, not a measured final-test failure. Means alone also do not establish statistical significance.')
evidence('src/aurora/training/metrics.py; assets/tables/diagnostic_metrics.csv; artifacts/phase3/selective_review/decision.json. Metric version: rgb-psnr-ssim-valid11-v1.')

page('7. Does the depth actually help?')
para('The trained RGB control tests whether the extra depth architecture improves reconstruction under matched per-run conditions. A second kind of experiment changes depth inside the already trained selective model. These interventions ask whether the output depends on the auxiliary signal and whether that dependence is helpful on particular scenes.')
sub('What the interventions change')
table([['Condition','Meaning'],['Native depth','Use the correct restored/normalized map for that input.'],['Zero depth','Provide a constant zero map while leaving the trained branch active. Its learned biases can still contribute.'],['Horizontally flipped depth','Keep RGB fixed but mirror the depth map, disrupting spatial correspondence.'],['Fusion removed','Actually disable depth fusion; this differs from merely passing zero depth.']],[125,385])
sub('Fixed three-case results for selective epoch 20')
table([['Case','Native depth PSNR / SSIM','Fusion removed PSNR / SSIM'],['Fish 14','26.284351 / 0.935168','22.127201 / 0.908886'],['Wreck 380','18.315401 / 0.869805','17.493034 / 0.869352'],['Equipment 729','20.422432 / 0.909081','21.338726 / 0.921352']],[110,200,200])
para('Removing fusion reduces both metrics for the fish and wreck, but improves both for the equipment image. For equipment, the improvement is +0.916294 dB PSNR and +0.012271 SSIM. Selective fusion therefore uses depth-related features, but has not removed harmful scene dependence.')
para('The flipped map is nearly tied with the correct map for the wreck. This cautions against interpreting every improvement as proof of accurate geometric reasoning. The interventions change the input distribution of one trained model and are limited to three existing diagnostic cases. They are not retrained controls and do not define a validated rule for deciding when to turn depth off.')
para('A defensible conclusion is: the depth branch influences reconstruction and can help or harm depending on the scene. The present experiment has not established a consistent depth benefit.')
evidence('artifacts/phase3/selective_review/; assets/evidence/selective_sensitivity.json; Docs/Selective Depth Fusion Results.md.')

page('8. MVP work: the complete user workflow')
picture('lifecycle.png',500,'Implemented operation lifecycle. Tokens tie results to inputs; cancellation terminates the worker and clears output.')
table([['Step','What the MVP does'],['Upload and validate','Accepts single-frame RGB JPEG/PNG, up to 20 MiB and 2.5 million decoded pixels. Handles orientation and colour profiles; rejects unsupported/malformed inputs.'],['Prepare the operation','Clears previous output/download state and assigns an operation identity. Loads the pinned checkpoint in a separate worker.'],['Estimate live depth','Runs the local frozen estimator for this image and restores an aligned normalized map. A depth error is reported as failure.'],['Enhance','Computes full-input RGB statistics, applies the trained model and removes replication padding.'],['Compare and export','Displays original and enhanced views and offers native-size lossless RGB PNG with an sRGB profile and source EXIF/GPS/thumbnail metadata removed.'],['Cancel or retry','Stops the active worker, clears the result and permits retry under a fresh operation identity. Late results from older operations are rejected.']],[125,385])
para('This is a local Gradio application backed by Python/PyTorch. The depth estimator uses locally available pinned model files. The export preserves the oriented input width and height. Depth preprocessing may resize internally, but enhancement geometry is restored before fusion and export.')
para('The worker isolates expensive model computation from the interface. A fresh worker/model instance is created for each operation. Session cleanup removes temporary processing files after unload or inactivity according to the application policy. These implementation choices are part of the measured runtime behaviour.')
evidence('src/aurora/app.py, worker.py, image.py and inference.py; configs/experimental_model.json.')

page('9. MVP evidence: what was checked and measured')
sub('Recorded functional checks')
para('The recorded fixture-test run passed 63 tests. The live localhost API workflow then processed two successive diagnostic uploads with actual depth estimation. Both output files reopened at native dimensions with sRGB profiles and source metadata removed. Output pixels changed, demonstrating that the workflow was not just previewing the input.')
para('The same live check observed cancellation during depth estimation, worker release, successful retry and clearing after a corrupt upload. Its recorded cancellation RPC took about 0.062 seconds; this single observation is not a cancellation latency guarantee. A later browser-export check confirmed that the downloaded fish result matched the packaged output pixel-for-pixel. These are functional checks, not independent human quality certification.')
sub('Repeated CPU operational profile')
rows=[['Input dimensions','n','Median s','p95 s','Max worker GB']]
for r in profile['summary']:
 rows.append([f"{r['width']} x {r['height']}",str(r['count']),f"{r['median_seconds']:.3f}",f"{r['p95_seconds']:.3f}",f"{r['max_worker_rss_bytes']/1e9:.3f}"])
table(rows,[130,30,105,105,140])
para('Hardware was a 12th Gen Intel Core i7-1255U running Linux and Python 3.14.8. Each of three diagnostic inputs had 3 warmups and 30 measured operations: 9 excluded warmups and 90 measured operations in total. Every operation loaded fresh models and ran live depth. Warmups warmed operating-system caches, not a permanently resident model.')
para('Median is the middle observed time. p95 is a percentile estimate over 30 observations per input, using linear interpolation. Peak RSS is the maximum resident memory observed for the worker, in decimal GB. The 90 measurements are repeated operations on three images, not 90 independent scenes.')
para('Timing covers the worker pipeline, including validation/process/model/depth/enhancement/export preparation. It excludes browser/upload transport and final PNG reopening. Memory excludes the UI/parent process. The 2.570 GB observation is not a worst-case memory limit for every allowed image. The results support a measured CPU prototype, with observed p95 times of approximately 3.4-8.4 seconds on these cases.')
evidence('artifacts/phase3/mvp/tests.log; assets/evidence/workflow.json and browser_output_parity.json; artifacts/phase3/mvp/profile/; profile_verification/.')

page('10. Show actual results during the explanation')
picture('before_after_14_img_.png',510,'Fish 14: raw input on the left, actual exported selective-depth MVP output on the right. The paired diagnostic score is 26.284351 dB / 0.935168 SSIM.')
para('Use this example to explain visible colour/contrast correction and native-size export. The image proves the implemented transformation produced an output; the numerical score measures agreement with the curated reference. It does not establish recovery of physically verified scene colour.')
picture('before_after_380_img_.png',510,'Wreck 380: raw input and actual output. Residual haze and reference-colour mismatch remain; native-depth score is 18.315401 dB / 0.869805 SSIM.')
para('Use the wreck to show a remaining limitation. The model changes appearance but does not solve severe degradation uniformly. Six input/output pairs and matched crops are available in assets/figures; their native originals and exported results are in assets/examples.')
evidence('assets/evidence/qualitative_provenance.json provides diagnostic membership, file hashes, crop coordinates and per-image scores.')

page('11. Current status and limits of the evidence')
table([['Area','Status to explain'],['Data and implementation','Audited partition, pinned models, geometry/colour handling, training and evaluation code exist.'],['Model experiments','RGB, scalar-depth and selective-depth 20-epoch runs completed; exact reload and recorded diagnostics available.'],['Experimental application','Selective-depth epoch 20 is active for integration/demo; live workflow and repeated CPU profile are recorded.'],['Depth-quality continuation','Failed: no declared candidate meets both matched RGB metrics under the continuation rule.'],['Formal artifact selection','RGB V3 failed the original screen. No formal depth artifact-selection approval is recorded.'],['Final confirmation','Paired final and challenging partitions remain sealed; no final benchmark results are reported.']],[135,375])
para('The original RGB artifact rubric required zero major cases and at most 4 minor cases out of 40. The recorded combined minor counts were 19, 34 and 31 for epochs 5, 10 and 20, with zero major cases reported. The two AI-assisted passes disagreed strongly at epoch 20: 3 versus 31 minor cases. The higher severity was retained; these are not two independent human certifications.')
para('Residual colour casts, haze, highlight fidelity and scene-dependent depth harm remain visible in the inspected cases. The 40 diagnostic images have influenced development, one primary seed was used, search budgets differ and source acquisition metadata is incomplete. These limits restrict generalization and comparative claims.')
sub('What completion would require')
para('A research-quality depth claim requires a justified experiment that improves the declared comparison and passes the recorded review criteria, followed by frozen-model final confirmation. Operational acceptance against the original broader device/resource contract also requires evidence beyond three profiled images. These are outstanding requirements, not completed results or scheduled new runs.')
para('The current milestone is a functional, profiled experimental MVP suitable for demonstrating the pipeline and reporting mixed development findings. It should not be described as a quality-approved final model or a completed final benchmark.')
evidence('assets/tables/rgb_artifact_review.csv; artifacts/phase2/evaluation_v3/; artifacts/phase3/selective_review/decision.json; artifacts/phase3/mvp/model_decision.json.')

page('12. A presentation sequence you can use')
sub('Opening: the problem and research question')
para('"Our project is Aurora, a local underwater image-enhancement system. Underwater photographs often have colour casts and weak contrast. We investigate whether relative depth estimated from the same photograph provides useful guidance beyond an RGB-only enhancer."')
sub('Explain the model')
para('"The trainable part is a compact residual U-Net. It learns a local RGB correction rather than predicting an unrelated image. A separate colour branch uses full-image RGB statistics. Depth Anything V2 Small stays frozen and supplies relative depth. A small depth encoder and selective gates combine this representation with RGB features at four scales."')
sub('Explain how it was trained and compared')
para('"We trained on 713 paired images using aligned 256 by 256 crops, an L1-plus-SSIM reconstruction objective and 20 epochs. We compared RGB-only, scalar-depth and selective-depth models under matched per-run settings. We used 40 diagnostic images repeatedly during development, so these are development results, not our final test."')
sub('Explain the MVP work')
para('"The application accepts a supported image, validates it, estimates depth live and runs enhancement in a cancellable worker. It displays the result and exports a native-size sRGB PNG. We checked cancellation, retry, stale-result clearing and downloaded output validity. Repeated CPU profiling used 30 operations on each of three images, after three warmups per image."')
sub('Give the result and limitation together')
para('"At epoch 20, selective depth achieved 20.9934 dB PSNR and 0.878263 SSIM, versus 20.9079 dB and 0.881212 for RGB. Depth improved the pixel-error metric slightly but reduced structural similarity. Interventions also showed that depth can help some scenes and harm others. We therefore have a working, measured prototype, but not a demonstrated consistent depth improvement."')
sub('A short live demonstration')
para('Start the app from Code with .venv/bin/python -m aurora app --port 7861, then open http://127.0.0.1:7861. Use permitted demo inputs. Show a normal enhancement and download, then cancellation/retry, and include a difficult example. Keep final/selection/challenging confirmation data out of the development demo. Use the saved image pairs and profiling table if the live application is not running.')
para('When asked about the current phase, say that depth experiments and MVP integration have been performed in Phase 3, while quality selection and the complete original acceptance gates remain unfinished. This avoids equating a working interface with a passed research result.')

page('13. Questions your supervisor may ask')
for q,a in [
('Why use depth?', 'It is a hypothesis about useful scene structure under spatially varying degradation. We test it against RGB-only training; current evidence is mixed, so the motivation is not proof.'),
('Is the whole system trained from scratch?', 'No. The enhancement generator is trained on the audited underwater pairs. Depth Anything V2 Small is pretrained, frozen and used as a feature supplier.'),
('Why U-Net and residual prediction?', 'U-Net combines coarse context with local skip features. Residual prediction anchors the mapping to the input. These are design choices; they do not guarantee preserved detail or zero artifacts.'),
('Is this segmentation or depth measurement?', 'The product task is RGB-to-RGB enhancement. Depth is an internal relative representation; the system does not provide calibrated object distances.'),
('Why is SSIM worse when PSNR is better?', 'They measure different aspects of reference agreement. The records establish the tradeoff, not its unique cause. It would require a separately controlled study to attribute it to a particular mechanism.'),
('Why use epoch 20 if epoch 10 has higher PSNR?', 'Epoch 20 is the predefined final checkpoint used for experimental integration, not a formal quality winner. Epoch 10 leads selective diagnostic PSNR; epoch 20 leads selective diagnostic SSIM. Neither passed continuation.'),
('Does turning depth off prove the RGB baseline is better?', 'No. Turning off fusion changes a depth-trained model at inference. The separately trained RGB model is the control; interventions test dependence and failure sensitivity.'),
('Can we claim state-of-the-art performance?', 'No matched external-method benchmark or final-set result has been produced here. The supported claims are implemented functionality, recorded development comparisons and measured CPU operation.'),
('What exactly is our deliverable now?', 'A local experimental depth-guided enhancer, trained model bundles, reproducible data/evaluation records, actual image pairs and an operational profile. Formal quality acceptance and final confirmation remain outstanding.'),
]:
 sub(q);para(a)

page('14. Vocabulary and evidence you can point to')
table([['Term','Meaning in this project'],['CNN / feature','Learned spatial filters / internal channel-valued image representations.'],['Encoder / decoder','Feature extraction at progressively coarser scales / reconstruction back toward image resolution.'],['Skip connection','A connection carrying an encoder feature to the matching decoder stage.'],['MLP','A small fully connected network; here it maps six RGB statistics to colour parameters.'],['Checkpoint / bundle','Saved learned parameters / parameters plus architecture, configuration and identity metadata.'],['Frozen / trainable','Weights kept fixed / weights updated by the reconstruction objective.'],['Ablation / sensitivity','A controlled component change / an intervention probing dependence; the three-case depth tests are inference interventions.'],['Diagnostic / final','Development data repeatedly inspected / reserved confirmation data after model choices are frozen.']],[125,385])
sub('Trace the explanation to its sources')
para('The original five-page team paper is in /home/shashwatsingh/Downloads/Aurora_Paper_IEEE.pdf. It supplies the research framing, not the latest completion status. The guide is checked against the following local implementation and records:','caption')
table([['Topic','Workspace record (relative to Aurora)'],['Architecture and depth','Code/src/aurora/models/; Code/src/aurora/depth.py'],['Training settings','Code/artifacts/phase3/selective_depth/config.json'],['Metrics and means','Code/src/aurora/training/metrics.py; Docs/Integration/assets/tables/diagnostic_metrics.csv'],['Sensitivity and decision','Code/artifacts/phase3/selective_review/'],['MVP functional checks','Code/artifacts/phase3/mvp/live_workflow.json; tests.log'],['Operational measurements','Code/artifacts/phase3/mvp/profile/ and profile_verification/'],['Active identity','Code/artifacts/phase3/mvp/model_decision.json'],['Actual image examples','Docs/Integration/assets/examples/ and assets/evidence/qualitative_provenance.json']],[125,385])
para('Active weights SHA-256: '+decision['weights_sha256'],'caption')
para('All implementation paths above are under Aurora/Code. The assets link in Docs/Integration resolves into Code/artifacts/paper_integration.','caption')

DOCS.mkdir(parents=True,exist_ok=True)
(DOCS/'Supervisor_Briefing.md').write_text('# Aurora: understanding and explaining the built system\n\n'+'\n'.join(md))
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#cbd5e1'));c.line(42,32,570,32)
 c.setFont('BriefSans',8);c.setFillColor(colors.HexColor('#64748b'))
 c.drawString(42,20,'Aurora | Supervisor briefing | Experimental MVP and development evidence')
 c.drawRightString(570,20,str(d.page))
SimpleDocTemplate(str(DOCS/'Aurora_Supervisor_Briefing.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=42,title='Aurora: Understanding and Explaining the Built System',author='Aurora project evidence').build(story,onFirstPage=footer,onLaterPages=footer)
sources=[Path('/home/shashwatsingh/Downloads/Aurora_Paper_IEEE.pdf'),ROOT/'src/aurora/models/rgb.py',ROOT/'src/aurora/models/scene_rgb.py',ROOT/'src/aurora/models/scene_depth.py',ROOT/'src/aurora/models/selective_depth.py',ROOT/'src/aurora/depth.py',ROOT/'src/aurora/training/metrics.py',ROOT/'artifacts/phase3/selective_depth/config.json',ROOT/'artifacts/phase3/mvp/model_decision.json',ASSETS/'tables/diagnostic_metrics.csv',ASSETS/'evidence/workflow.json',ROOT/'artifacts/phase3/mvp/profile/summary.json',ROOT/'artifacts/phase3/selective_review/decision.json',ROOT/'artifacts/phase3/mvp/tests.log']
(ASSETS/'evidence/supervisor_briefing_sources.json').write_text(json.dumps({'source_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]},indent=2)+'\n')
print('Created supervisor briefing PDF and editable Markdown.')
