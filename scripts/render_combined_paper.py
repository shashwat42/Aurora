"""Render the editable combined-paper Markdown using the established layout."""
from pathlib import Path
import re, sys
from xml.sax.saxutils import escape
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'Docs/Combined Paper'
TITLE = 'Relative-Depth-Guided Residual Learning for Underwater Image Enhancement'
COMPACT = '--six-pages' in sys.argv
STEM = TITLE + (' - 6 pages' if COMPACT else '')
s = (OUT / (STEM + '.md')).read_text()
AUTHORS = s.splitlines()[2]
AFF = s.splitlines()[4]
# Use the existing Python 3.14 project environment with bundled pure-Python ReportLab.
sys.path.append('/home/shashwatsingh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,NextPageTemplate,KeepTogether,BalancedColumns,FrameBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from PIL import Image as PILImage
for name,file in [('Paper','Regular'),('PaperBold','Bold'),('PaperItalic','Italic')]:pdfmetrics.registerFont(TTFont(name,f'/usr/share/fonts/liberation-serif-fonts/LiberationSerif-{file}.ttf'))
pdfmetrics.registerFont(TTFont('MathSymbols','/usr/share/fonts/google-noto/NotoSansMath-Regular.ttf'))
pdfmetrics.registerFontFamily('Paper',normal='Paper',bold='PaperBold',italic='PaperItalic',boldItalic='PaperBold')
body=ParagraphStyle('body',fontName='Paper',fontSize=9.4,leading=11.7,alignment=TA_JUSTIFY,spaceAfter=6,allowWidows=0,allowOrphans=0)
styles={'body':body,'h':ParagraphStyle('h',parent=body,fontName='PaperBold',fontSize=10,leading=12,spaceBefore=10,spaceAfter=6,alignment=TA_CENTER,keepWithNext=True),'sub':ParagraphStyle('sub',parent=body,fontName='PaperItalic',spaceBefore=6,keepWithNext=True),'cap':ParagraphStyle('cap',parent=body,fontSize=8.3,leading=10.1,alignment=0,spaceAfter=8),'ref':ParagraphStyle('ref',parent=body,fontSize=8.1,leading=9.7,alignment=0),'eq':ParagraphStyle('eq',parent=body,fontSize=8.5,leading=11.6,alignment=TA_CENTER,spaceBefore=5,spaceAfter=8),'cell':ParagraphStyle('cell',parent=body,fontSize=8,leading=9.5,alignment=0,spaceAfter=0)}
# Explicit typesetting equivalents of the four source display equations.
styles['tablecap'] = ParagraphStyle('tablecap', parent=styles['cap'], keepWithNext=True)
if COMPACT:
    body.fontSize = 10.2
    body.leading = 12.6
    body.spaceAfter = 4
    styles['ref'].fontSize = 9.0
    styles['ref'].leading = 10.6
    styles['ref'].spaceAfter = 4
    styles['ref'].alignment = TA_JUSTIFY
eqs=[
'J<sub>pre</sub> = I + 0.5 tanh(G<sub>θ</sub>(I,D))<br/>+ Δg(s) ⊙ I + b(s),<br/>J<sub>out</sub> = clip(J<sub>pre</sub>,0,1). &nbsp; (1)',
'D(p) = [d(p) - d<sub>min</sub>] / [d<sub>max</sub> - d<sub>min</sub>]<br/>if d<sub>max</sub> - d<sub>min</sub> &gt; 10<super>-6</super>;<br/>D(p) = 0 otherwise. &nbsp; (2)',
'Z<sub>l</sub> = P<sub>l</sub>(F<sub>D</sub><super>l</super>),<br/>S<sub>l</sub> = σ(W<sub>2,l</sub> ReLU(W<sub>1,l</sub>[F<sub>RGB</sub><super>l</super>, Z<sub>l</sub>])),<br/>F<sub>fused</sub><super>l</super> = F<sub>RGB</sub><super>l</super> + tanh(a<sub>l</sub>) S<sub>l</sub> ⊙ Z<sub>l</sub>. &nbsp; (3)',
'L<sub>rec</sub> = λ<sub>1</sub> ||J<sub>pre</sub> - J||<sub>1</sub><br/>+ λ<sub>s</sub> [1 - SSIM(J<sub>out</sub>, J)]. &nbsp; (4)']
eqiter=iter(eqs)
s=re.sub(r'(?s)\\\[.*?\\\]',lambda m:'@@EQ'+str(eqs.index(next(eqiter)))+'@@',s)
inline={r'I\in[0,1]^{H\times W\times3}':'I ∈ [0,1]<super>H×W×3</super>',r'J':'J',r'D=\mathcal{N}(\mathcal{R}(D_\psi(I)))':'D = N(R(D<sub>ψ</sub>(I)))',r's':'s',r'd':'d',r'P_l(F_D^l)':'P<sub>l</sub>(F<sub>D</sub><super>l</super>)',r'\tanh(a_l)':'tanh(a<sub>l</sub>)',r'S_l=0.5':'S<sub>l</sub> = 0.5',r'S_l=1':'S<sub>l</sub> = 1',r'\lambda_1=1.0':'λ<sub>1</sub> = 1.0',r'\lambda_s=0.2':'λ<sub>s</sub> = 0.2',r'\widetilde{J}':'J<sub>pre</sub>',r'2\times10^{-4}':'2 × 10<super>-4</super>',r'10^{-4}':'10<super>-4</super>',r'K_1=0.01':'K<sub>1</sub> = 0.01',r'K_2=0.03':'K<sub>2</sub> = 0.03'}
eqs=[t.replace('⊙','<font name="MathSymbols">⊙</font>') for t in eqs]
inline={k:v.replace('∈','<font name="MathSymbols">∈</font>') for k,v in inline.items()}
def fmt(t):
    t=escape(t)
    t=re.sub(r'\\\((.*?)\\\)',lambda m:inline[m[1]],t)
    t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
    t=re.sub(r'`(.*?)`',r'\1',t)
    t=re.sub(r'https?://[^\s]+',lambda m:'<link href="'+m[0]+'" color="#17385b">'+m[0]+'</link>',t)
    return t
W,H=612,792; M=40; GAP=20; CW=(W-2*M-GAP)/2
frames=[Frame(M,40,CW,H-80,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(M+CW+GAP,40,CW,H-80,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
firstframes=[Frame(M,H-118,W-2*M,78,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(M,40,CW,H-174,id='firstleft',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(M+CW+GAP,40,CW,H-174,id='firstright',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
def footer(c,d):
    c.setFont('Paper',8);c.drawCentredString(W/2,23,str(d.page))
def architecture_page(c,d):
    footer(c,d)
    path=OUT/'../Integration/assets/figures/architecture.png'
    with PILImage.open(path) as im:iw,ih=im.size
    width=480; height=width*ih/iw
    c.drawImage(str(path),(W-width)/2,H-40-height,width=width,height=height,mask='auto')
    cap=re.search(r'!\[[^\n]*\]\([^\n]*architecture\.png\)\n\n([^\n]+)',s)[1]
    p=Paragraph(fmt(cap),styles['cap']);_,ph=p.wrap(W-2*M,60)
    p.drawOn(c,M,H-40-height-6-ph)
doc=BaseDocTemplate(str(OUT/(STEM+'.pdf')),pagesize=(W,H),title=TITLE,author=AUTHORS,leftMargin=M,rightMargin=M,topMargin=40,bottomMargin=40)
doc.addPageTemplates([PageTemplate('paper',[Frame(M,40,W-2*M,H-80,id='paper',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer)])
if COMPACT:
    doc.pageTemplates.clear()
    title_frame = Frame(M,H-150,W-2*M,110,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    body_frames = [Frame(x,40,CW,H-174,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for x,name in [(M,'firstleft'),(M+CW+GAP,'firstright')]]
    architecture_frames=[Frame(x,40,CW,H-80-380,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for x,name in [(M,'archleft'),(M+CW+GAP,'archright')]]
    doc.addPageTemplates([PageTemplate('first',[title_frame]+body_frames,onPage=footer),PageTemplate('architecture',architecture_frames,onPage=architecture_page,autoNextPageTemplate='columns'),PageTemplate('columns',frames,onPage=footer)])
story=[Paragraph(escape(TITLE),ParagraphStyle('title',fontName='PaperBold',fontSize=18,leading=20,alignment=1,spaceAfter=9)),Paragraph(escape(AUTHORS),ParagraphStyle('authors',fontName='Paper',fontSize=10,leading=12,alignment=1)),Paragraph(escape(AFF),ParagraphStyle('aff',fontName='Paper',fontSize=8.5,leading=11,alignment=1)),Spacer(1,16)]
content=s[s.index('## Abstract'):]
if COMPACT:
    story.extend([NextPageTemplate('architecture'),FrameBreak])
blocks=re.split(r'\n\s*\n',content.strip())
columns=[]
full=False
def flush_columns():
    if columns:
        if COMPACT:
            story.extend(columns)
        else:
            story.append(BalancedColumns(columns[:],nCols=2,needed=72,leftPadding=0,rightPadding=0,innerPadding=GAP/2,topPadding=0,bottomPadding=0,spaceAfter=8))
        columns.clear()
i=0
while i<len(blocks):
    b=blocks[i]
    if b.startswith('## Appendix A'):
        flush_columns();story.append(PageBreak());full=True
    elif b.startswith('## Appendix B'):
        story.append(PageBreak())
    if b=='## 3. Methodology':
        flush_columns()
        story.append(Paragraph('3. Methodology',styles['h']))
    elif b.startswith('!['):
        if not COMPACT:
            flush_columns()
        path=re.search(r'\]\((.*?)\)',b)[1];path=(OUT/path).resolve()
        if COMPACT and path.name=='architecture.png':
            i+=2
            continue
        with PILImage.open(path) as im:iw,ih=im.size
        width=480 if path.name=='architecture.png' else 450
        if 'native_' in path.name:width=370
        if path.name=='lifecycle.png':width=480
        if COMPACT:width=CW
        img=Image(str(path),width=width,height=width*ih/iw)
        caption=Paragraph(fmt(blocks[i+1]),styles['cap'])
        (columns if COMPACT else story).append(KeepTogether([img,Spacer(1,5),caption]))
        i+=1
    elif b.startswith('#'):
        level=len(b)-len(b.lstrip('#'));text=b.lstrip('# ').strip()
        flow=Paragraph(fmt(text),styles['h' if level==2 else 'sub'])
        if level>=3 and not full and i+1<len(blocks) and not blocks[i+1].startswith(('#','![','|','@@')):
            columns.append(Paragraph('<i>'+fmt(text)+'</i><br/>'+fmt(blocks[i+1].replace('\n',' ')),styles['body']))
            i+=1
        else:
            (story if full else columns).append(flow)
    elif b.startswith('@@EQ'):
        columns.append(Paragraph(eqs[int(re.search(r'\d+',b)[0])],styles['eq']))
    elif b.startswith('|'):
        data=[[Paragraph(fmt(c.strip()),styles['cell']) for c in line.strip('|').split('|')] for line in b.splitlines() if not re.match(r'^\|[\s:|-]+\|$',line)]
        n=len(data[0]);widths=([95,35,CW-130] if n==3 else [48]+[(CW-48)/(n-1)]*(n-1))
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=0)
        tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
        if columns and isinstance(columns[-1], Paragraph) and columns[-1].style.name == 'tablecap':
            group = Table([[columns.pop()], [tab]], colWidths=[CW], splitByRow=0)
            group.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
            columns.extend([group, Spacer(1,8)])
        else:
            columns.extend([tab,Spacer(1,8)])
    else:
        style='tablecap' if b.startswith('Table ') else 'cap' if b.startswith('Fig.') else 'ref' if re.match(r'^\[(?:E)?\d+\]',b) else 'body'
        paragraph = Paragraph(fmt(b.replace('\n',' ')),styles[style])
        (story if full else columns).append(KeepTogether([paragraph]) if COMPACT and style=='ref' else paragraph)
    i+=1
flush_columns()
doc.build(story)
print(OUT/(STEM+'.pdf'))
