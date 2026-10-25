"""Воспроизводимое оформление Markdown-материалов в PDF (ReportLab)."""
from html import escape
from pathlib import Path
import re
import sys
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, KeepTogether)

FONT = Path("C:/Windows/Fonts")
for name, filename in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                       ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf"),
                       ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
TEAL, INK, GREY = colors.HexColor("#087e8b"), colors.HexColor("#17343c"), colors.HexColor("#62767d")
BODY = ParagraphStyle("Body", fontName="Arial", fontSize=9.2, leading=13.6,
                      spaceAfter=7, textColor=INK, wordWrap="CJK", allowWidows=0, allowOrphans=0)
HEAD = {n: ParagraphStyle(f"h{n}", parent=BODY, fontName="Arial-Bold",
                         fontSize={1:18,2:13,3:10.5}.get(n,10),
                         leading={1:23,2:17,3:14}.get(n,14),
                         textColor=TEAL if n<3 else INK, spaceBefore=13, spaceAfter=8,
                         keepWithNext=True) for n in range(1,5)}
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.1, leading=11.1, spaceAfter=0)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Mono", fontSize=7.1, leading=10,
                      backColor=colors.HexColor("#f2f5f6"), borderPadding=6)


def markup(text):
    text=text.replace("\u2011","-").replace("\u2010","-")
    text=escape(text)
    text=re.sub(r"\[([^\]]+)\]\((https?://[^ )]+)\)", r'<link href="\2" color="#087e8b">\1</link>',text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`([^`]+)`",r'<font name="Mono">\1</font>',text)
    return text


class Document(BaseDocTemplate):
    def __init__(self,path,title):
        super().__init__(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=47,bottomMargin=43,
                         title=title,author="Черных Юрий; Бурундуков Михаил — учебный проект")
        self.title_text=title
        self.heading_count=0
        frame=Frame(42,43,A4[0]-84,A4[1]-90,id="body")
        self.addPageTemplates(PageTemplate(id="main",frames=frame,onPage=self.decorate))
    def decorate(self,canvas,doc):
        canvas.saveState();canvas.setFont("Arial",7.3);canvas.setFillColor(GREY)
        canvas.drawString(42,A4[1]-27,"Аудиомодуль ВКР · ОКБ №1 · учебный прототип")
        canvas.setStrokeColor(colors.HexColor("#dce5e7"));canvas.line(42,A4[1]-33,A4[0]-42,A4[1]-33)
        canvas.drawString(42,24,"Подготовлено 24–25 октября 2026 · фактические результаты и ограничения")
        canvas.drawRightString(A4[0]-42,24,str(doc.page));canvas.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in {"h1","h2"}:
            key=f"heading-{self.heading_count}";self.heading_count+=1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0 if flowable.style.name=="h1" else 1,closed=False)


def table(rows,width):
    count=max(map(len,rows));rows=[r+[""]*(count-len(r)) for r in rows]
    weights=[]
    for col in range(count):
        lengths=sorted(len(re.sub(r'\*\*|`','',r[col])) for r in rows)
        weights.append(max(12,min(65,lengths[len(lengths)//2])))
    widths=[width*w/sum(weights) for w in weights]
    parsed=[[Paragraph(markup(cell),CELL) for cell in row] for row in rows]
    out=Table(parsed,colWidths=widths,repeatRows=1,hAlign="LEFT")
    out.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6f1f2")),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#b7c8cc")),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    return out


def parse_markdown(text,width):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("```"):
            code=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith("```"):
                code.extend(textwrap.wrap(lines[i],width=92,replace_whitespace=False,drop_whitespace=False) or [""])
                i+=1
            out.append(Paragraph("<br/>".join(escape(x).replace(" ","&#160;") for x in code),CODE));i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+",c) for c in cells):rows.append(cells)
                i+=1
            out.append(table(rows,width));out.append(Spacer(1,9));continue
        heading=re.match(r"^(#{1,4})\s+(.+)",line)
        if heading:
            out.append(Paragraph(markup(heading[2]),HEAD[len(heading[1])]))
            i+=1;continue
        if line=="---":out.append(Spacer(1,9));i+=1;continue
        if line.startswith("- ") or re.match(r"^\d+\. ",line):
            out.append(Paragraph(markup(line),BODY));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|- |\d+\. )",lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        out.append(Paragraph(markup(" ".join(paragraph)),BODY))
    return out


def build(output,title,sources):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if "план" in title.lower():
        BODY.leading=13.0
        BODY.spaceAfter=6
        for style in HEAD.values():style.spaceBefore=11
    story=[]
    for index,path in enumerate(sources):
        if index:story.append(PageBreak())
        text=Path(path).read_text(encoding="utf-8-sig")
        if Path(path).name=="manual-tests.md":
            for block in re.split(r"(?m)(?=^## M\d{2})",text):
                flows=parse_markdown(block,A4[0]-84)
                if block.startswith("## M"):story.append(KeepTogether(flows))
                else:story.extend(flows)
        else:
            story.extend(parse_markdown(text,A4[0]-84))
    Document(output,title).build(story)
    print(output)


if __name__=="__main__":
    build(sys.argv[1],sys.argv[2],sys.argv[3:])
