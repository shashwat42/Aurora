"""Create the team-facing integration brief from verified MVP assets."""
import csv,json,shutil,sys
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
sys.path.append('/home/shashwatsingh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("AuroraSans", "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("AuroraSans-Bold", "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"))
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter

DOCS=ROOT.parent/'Docs/Integration';ASSETS=ROOT/'artifacts/paper_integration';F=ASSETS/'figures'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyA',fontName='AuroraSans',fontSize=9.5,leading=13,spaceAfter=8))
styles.add(ParagraphStyle(name='CaptionA',fontName='AuroraSans',fontSize=8,leading=11,textColor=colors.HexColor('#475569'),spaceAfter=9))
styles.add(ParagraphStyle(name='TitleA',fontName='AuroraSans-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#0f172a'),spaceAfter=15))
styles.add(ParagraphStyle(name='HeadA',fontName='AuroraSans-Bold',fontSize=15,leading=19,textColor=colors.HexColor('#0f172a'),spaceAfter=12))


def p(s,style='BodyA'):return Paragraph(escape(s),styles[style])


def table(data,widths):
    t=Table([[p(str(c),'CaptionA') for c in row] for row in data],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2e8f0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#94a3b8')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')])]))
    return t


def image(path,w):
    from PIL import Image as PI
    with PI.open(path) as im:h=w*im.height/im.width
    return Image(str(path),width=w,height=h)


def main():
    metrics=list(csv.DictReader((ASSETS/'tables/diagnostic_metrics.csv').open()))
    profile=list(csv.DictReader((ASSETS/'tables/runtime_profile.csv').open()))
    DOCS.mkdir(exist_ok=True,parents=True)
    caption='''# Figure captions and placement

All image results below use the frozen selective-depth epoch 20 MVP (`b59fdab3...`), unless labelled RGB20. All scene examples are from the diagnostic split and were already used during development. They are illustrative cases, not an independent qualitative benchmark. Raw and enhanced files retain original native dimensions; comparison-sheet thumbnails are resized uniformly for display only.

| Asset in assets/figures | Suggested caption / use |
| --- | --- |
| architecture.svg / .png | Implemented Aurora selective-depth MVP: frozen relative-depth estimation, aligned depth/RGB encoding, spatial/channel modulation and input-statistics colour correction. Gates are not calibrated confidence. Replace existing Fig. 1. |
| lifecycle.svg / .png | Operation-token worker lifecycle; cancellation clears output and stops the worker. Use in application section or supplement. |
| diagnostic_psnr.svg / .png | Mean native diagnostic 40 PSNR at the three declared checkpoints. Each point averages the same 40 images; one primary seed; no final-test claim. |
| diagnostic_ssim.svg / .png | Mean native diagnostic 40 SSIM. Depth models remain below their matched RGB controls; report alongside PSNR. |
| runtime_latency.svg / .png | p95 worker-pipeline time for three specific diagnostic inputs, each 30 measurements after 3 warmups on Intel i7-1255U; live depth and fresh model instances every operation. Excludes browser/upload transport. |
| runtime_memory.svg / .png | Maximum worker RSS over the same30 measured operations per input, decimal GB. Excludes UI/parent RSS; not a universal maximum-memory guarantee. |
| before_after_14_img_.png | Raw fish input and actual exported depth-MVP result. Visible colour/contrast correction demonstrates operation, not recovery of verified physical detail. |
| before_after_887_img_.png | Raw diver/coral input and actual MVP output. Remaining colour differences should be considered against the curated reference in supplementary comparisons. |
| before_after_284_img_.png | Low-resolution diagnostic input and its actual enhanced output, preserved at native dimensions. |
| before_after_380_img_.png | Hazy wreck input/output. Green haze and reference-colour mismatch persist, illustrating a limitation. |
| before_after_729_img_.png | Equipment input/output. This case also shows harmful depth dependence in intervention tests; do not present it as universal depth improvement. |
| before_after_851_img_.png | Reef/hand input/output. Pale/pink highlights remain a fidelity concern. |
| qualitative_whole_1.png / _2.png | Context panels ordered raw, trained RGB20, depth MVP20, curated reference. PSNR annotations use native float predictions, not resized display images. |
| qualitative_native_1.png / _2.png | Fixed same-coordinate native-pixel crops from the above cases. Crop coordinates and source/output hashes are in evidence/qualitative_provenance.json. |

The browser-exported fish output was reopened and is pixel-identical to the packaged result; verification is recorded in evidence/browser_output_parity.json.
'''
    (DOCS/'03_Figure_Captions.md').write_text(caption)
    source=(ROOT.parent/'Docs/Research Paper Draft.md').read_text()
    source=source.replace(' A concatenation-only depth architecture was discussed in the original design but has not been trained/evaluated here.', '').replace('The user authorized an experimental prototype despite the retained rejection.', 'The deployed prototype retains the recorded artifact rejection.')
    results=source[source.index('## 7. Development Results'):source.index('## 8. Limitations')]
    methods=source[source.index('## 3.'):source.index('## 4.')] if '## 3.' in source else ''
    abstract='''# Ready-to-integrate text

## Revised abstract draft

Aurora is a local, native-resolution underwater image-enhancement prototype combining a residual U-Net, an input-statistics colour branch and frozen monocular relative depth. Scalar and selective multiscale fusion variants were trained with raw L1 and bounded SSIM objectives. On 40 repeatedly inspected diagnostic images, both depth architectures increased PSNR but decreased SSIM relative to matched RGB checkpoints. At epoch 20, selective depth achieved 20.9934 dB / 0.878263 SSIM, compared with 20.9079 dB / 0.881212 for RGB. Depth interventions revealed useful and harmful case dependence; no final depth-benefit claim is made. The experimental MVP performs live depth estimation, interruptible enhancement and native-size sRGB PNG export. On an Intel i7-1255U, 30 measured operations per diagnostic input after 3 warmups yielded p95 worker-pipeline times of 3.360, 6.314 and 8.361 seconds for 276×182, 1280×803 and 1800×1350 images. These measurements exclude browser/upload transport. The system demonstrates a functional profiled prototype while retaining failed quality gates, reviewer uncertainty and sealed final evaluation partitions.

The sections below describe the implementation and development findings.

'''
    (DOCS/'02_Ready_to_Integrate.md').write_text(abstract+methods+'\n'+results)
    (DOCS/'Evidence_Based_Working_Draft.md').write_text(source)
    readme='''# Aurora paper integration package

This package documents the implemented MVP, recorded development experiments and measured operation profile.

## Contents

- `Aurora_Supervisor_Briefing.pdf`: plain-language explanation of the model, experiments and MVP, with speaking notes and supervisor questions.
- `Supervisor_Briefing.md`: editable version of the supervisor briefing.

- `Aurora_MVP_Integration_Brief.pdf`: model overview, measured results, architecture and input/output comparisons.
- `02_Ready_to_Integrate.md`: replacement abstract/methods/results text.
- `03_Figure_Captions.md`: captions, limitations and placement.
- `Evidence_Based_Working_Draft.md`: current project draft with verified results.
- `assets/figures/`: 6 editable SVG diagrams/charts, matching high-resolution PNGs,6 raw/output pairs and 4 context/native sheets.
- `assets/examples/`: 6 cases with original raw file, native MVP output, RGB control output and reference display.
- `assets/tables/`: native diagnostic metrics, measured runtime profile and RGB artifact screen CSVs.
- `assets/evidence/`: provenance, checksums and verified experiment records.

`assets` is a relative symlink to `Code/artifacts/paper_integration/`, keeping implementation/non-document assets inside Code. A portable ZIP with the link materialized is supplied in Code/artifacts/paper_integration_bundle.zip; its internal Integration folder is self-contained.

## Evidence boundary

All qualitative examples and PSNR/SSIM means are diagnostic development results. No final-test/challenging or new selection images were evaluated. RGB formal artifact approval failed and depth diagnostic continuation failed. The active selective-depth checkpoint is deployed experimentally and has not passed formal quality selection. Cumulative architecture-search budgets differ. The operational measurements cover three specific inputs and worker-only resource use.

Two depth architectures and one V3 RGB model each used 20 epochs, seed 20261101 and three checkpoints; earlier exploratory work remains in the research history. The measured operational profile has 90 recorded operations / 9 warmups, not 90 independent images. The image comparisons include useful corrections and remaining defects.

Dataset images and model assets retain their source restrictions. This is an internal team integration package; check UIEB/source permissions before public redistribution. No model weights or environments are included. Figure generators are in Code/scripts/build_paper_assets.py and build_integration_brief.py; they use project Python 3.14 with bundled pure-Python ReportLab, no additional venv.
'''
    (DOCS/'README.md').write_text(readme)
    overview_rows = [
        ['Item', 'Current MVP'],
        ['Task', 'Single-image underwater RGB enhancement at native resolution.'],
        ['Model family', 'Deep learning: convolutional neural network (CNN), using a depth-guided residual U-Net.'],
        ['Enhancement model', 'Selective multiscale depth fusion plus an RGB-statistics colour-correction MLP; 517,913 trainable generator parameters.'],
        ['Depth estimator', 'Pretrained Depth Anything V2 Small, frozen during enhancement training; estimates relative depth, not physical distance.'],
        ['Active checkpoint', 'Selective-depth epoch 20; experimental MVP checkpoint, not quality-selected.'],
        ['Training objective', 'Raw prediction L1 loss + 0.2 x (1 - bounded SSIM). '],
        ['PSNR (higher is better)', 'Pixel fidelity against the curated reference, in decibels. Current mean: 20.9934 dB.'],
        ['SSIM (higher is better)', 'Local structure, contrast and brightness agreement with the reference. Current mean: 0.878263.'],
        ['Metric protocol', 'Native float RGB, data range 1; equal weight per image. SSIM: 11 x 11 Gaussian valid window, sigma 1.5. Means use 40 repeatedly inspected diagnostic images.'],
        ['Operational metrics', 'Median and p95 worker-pipeline latency; maximum observed worker RSS. Three inputs, 30 recorded operations each, after 3 warmups each; excludes browser/upload transport.'],
        ['Visual assessment', 'Check colour shifts, highlights, texture and border defects alongside numerical fidelity.'],
    ]
    overview_md = '## MVP model and evaluation at a glance\n\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in [overview_rows[0], ['---', '---']] + overview_rows[1:]) + '\n\nThese are development results, not final-test benchmarks. Depth improves PSNR but lowers SSIM relative to the matched RGB checkpoint; no consistent depth-benefit claim follows.\n\n'
    for name in ['README.md', '02_Ready_to_Integrate.md', 'Evidence_Based_Working_Draft.md']:
        path = DOCS/name
        heading, rest = path.read_text().split('\n', 1)
        path.write_text(heading + '\n\n' + overview_md + rest.lstrip('\n'))
    story=[p('Aurora','TitleA'),p('MVP model and evaluation at a glance','HeadA'),table(overview_rows,[130,380]),Spacer(1,10),p('Development evidence only: these diagnostic images were repeatedly inspected. Depth improves PSNR but lowers SSIM relative to the matched RGB checkpoint. No final-test benchmark or consistent depth-benefit claim is established.','CaptionA'),PageBreak()]
    story += [p('Implemented system','HeadA'),image(F/'architecture.png',510),p('Figure A. Actual selective-depth MVP architecture. Local residual bound is ±0.5; colour gain/offset broaden total raw capacity. Generator: 517,913 parameters; frozen estimator excluded. Gates are feature controls, not calibrated confidence.','CaptionA'),p('The native input feeds both frozen depth estimation and RGB features. Full-image RGB mean/std also supply the colour branch during crop training. Live UI operations run depth estimation explicitly; no RGB fallback is substituted on depth failure.'),image(F/'lifecycle.png',510),p('Figure B. Operation-token lifecycle. Existing API tests observed depth-stage cancellation, worker release, retry and invalid-upload clearing. The browser capture also verified an actual PNG download and cancellation.','CaptionA'),PageBreak()]
    matrix=[['Epoch','RGB PSNR / SSIM','Scalar depth','Selective depth']]
    for epoch in [5,10,20]:matrix.append([epoch]+[f"{float(r['psnr']):.6f} / {float(r['ssim']):.6f}" for r in metrics if int(r['epoch'])==epoch])
    story += [p('Native diagnostic fidelity','HeadA'),table(matrix,[45,155,155,155]),Spacer(1,10),image(F/'diagnostic_psnr.png',510),image(F/'diagnostic_ssim.png',510),p('Figures C-D. Each point is a mean over the same 40 native diagnostic images. Depth raises PSNR but lowers SSIM at each matched checkpoint. These are adaptive development results from one primary seed, not final-test results or statistical superiority evidence.','CaptionA'),PageBreak()]
    matrix=[['Native input','n','Median (s)','p95 (s)','Max worker GB']]
    for r in profile:matrix.append([r['width']+' x '+r['height'],r['count'],f"{float(r['median_seconds']):.3f}",f"{float(r['p95_seconds']):.3f}",f"{int(r['max_worker_rss_bytes'])/1e9:.3f}"])
    story += [p('Measured operational profile','HeadA'),p('Intel Core i7-1255U | Python 3.14.8 | Linux 7.2.13 / glibc 2.43. Three warmups then 30 measured operations per input; 90 measured and 9 excluded warmups. Every operation uses live depth and a fresh worker/model instance.'),table(matrix,[150,35,105,105,115]),Spacer(1,8),image(F/'runtime_latency.png',510),image(F/'runtime_memory.png',510),p('Figures E-F. Timing includes validation/process/model/depth/enhancement/export work, but excludes browser/network upload transport. RSS is worker-only; GB is decimal. Three specific images do not establish worst-case resolution-class limits.','CaptionA'),PageBreak()]
    for page in [1,2]:
        story += [p('Actual image comparisons: '+('correction examples' if page==1 else 'remaining limitations'),'HeadA'),image(F/f'qualitative_whole_{page}.png',470),p('Raw input | RGB20 control | actual selective-depth MVP20 output | curated reference. Overview images are resized only for layout. PSNR annotations use native float evaluation; output files retain native dimensions.','CaptionA'),image(F/f'qualitative_native_{page}.png',380),p('Matched native-pixel crops; coordinates/source hashes are supplied in evidence/qualitative_provenance.json. Layout scales the sheet for this PDF; use external PNGs/native files for detailed inspection. All cases are diagnostic and previously inspected.','CaptionA'),PageBreak()]
    story += [p('Before and after: actual exported results','HeadA'),image(F/'before_after_14_img_.png',510),p('Figure G. Raw fish photograph and the actual exported depth-MVP output. The browser-downloaded file was reopened and matches the packaged enhanced image pixel-for-pixel. This demonstrates the implemented transform/export workflow.','CaptionA'),image(F/'before_after_380_img_.png',510),p('Figure H. Hazy wreck: correction occurs, but residual green haze and reference-colour mismatch remain. Six input/output pairs and all full-resolution files are supplied.','CaptionA'),PageBreak()]
    story += [p('Interpretation and insertion checklist','HeadA'),p('What the evidence supports','HeadA'),p('A functional experimental depth-guided MVP with native export, explicit live depth, cancellation/retry and a measured CPU profile. Recorded development experiments show a PSNR/SSIM tradeoff and scene-dependent depth use.'),p('Recorded limitations','HeadA'),p('RGB artifact review recorded 19/34/31 minor cases out of 40 at epochs 5/10/20 and failed its approval criterion. Selective depth failed the diagnostic continuation criterion. Final evaluation partitions remain unopened; architecture-search budgets differ and operational profiling covers three inputs.'),table([['Paper section','Insert / replace'],['Abstract / III methods','Use the current colour branch and selective fusion formula; avoid claiming proven depth benefit.'],['V setup / VI evaluation','Completed models, parameter counts, data splits and honest search-budget history.'],['VIII results','Diagnostic table, PSNR/SSIM charts, runtime table and before/after examples.'],['IX discussion','Mixed dependence, equipment failure and original artifact-review disagreement.'],['X conclusion','Functional profiled prototype; final confirmation remains unperformed.']],[120,390]),Spacer(1,14),p('Package navigation','HeadA'),p('02_Ready_to_Integrate.md contains editable methods and results; 03_Figure_Captions.md supplies captions. assets/figures contains SVG/PNG; assets/examples contains raw/exported images; assets/tables and assets/evidence preserve numerical sources and provenance.'),p('Full operational and quality records stay in the workspace. Use the portable ZIP when sharing internally so relative asset links remain valid. Check source-image redistribution terms before external publication.','CaptionA')]
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'));canvas.line(42,31,570,31)
        canvas.setFont('AuroraSans',8);canvas.setFillColor(colors.HexColor('#64748b'));canvas.drawString(42,19,'Aurora | Experimental MVP evidence | Diagnostic results, not final confirmation');canvas.drawRightString(570,19,str(doc.page))
    SimpleDocTemplate(str(DOCS/'Aurora_MVP_Integration_Brief.pdf'),pagesize=letter,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=42,title='Aurora MVP Findings and Paper Integration',author='Aurora project evidence package').build(story,onFirstPage=footer,onLaterPages=footer)
    print('Integration brief PDF and editable team documents generated.')


if __name__=='__main__':main()
