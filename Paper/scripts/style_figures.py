"""Replace development-round labels in existing vector charts; retain all geometry/data.
Requires CairoSVG only when regenerating assets.
Code edited with OpenAI Codex (GPT-6); release date unavailable.
"""
from pathlib import Path
import hashlib,json
import cairosvg
P=Path(__file__).resolve().parents[1];R=P.parent
src=R/'问题2/docs/paper_current/figures'
records=[]
for old,new in [('r4_r4_03_confusion','q2_confusion'),('r4_r4_04_missing','q2_missing')]:
 original=src/(old+'.svg');text=original.read_text()
 for a,b in [('Round 3 reference','Unweighted'),('Frozen round 4','Class-balanced'),('Round 3','Unweighted'),('Round 4','Class-balanced')]:text=text.replace(a,b)
 svg=P/'assets'/(new+'.svg');pdf=svg.with_suffix('.pdf');svg.write_text(text)
 cairosvg.svg2pdf(bytestring=text.encode(),write_to=str(pdf))
 records.append({'source':original.relative_to(R).as_posix(),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'svg':svg.relative_to(P).as_posix(),'svg_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'pdf':pdf.relative_to(P).as_posix(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'change':'round labels only; numeric text, paths and chart geometry preserved'})
(P/'assets/figure_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
