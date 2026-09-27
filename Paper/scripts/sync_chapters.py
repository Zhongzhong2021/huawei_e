"""Render edited manuscripts; never import over the curated Markdown sources.
Build-script editing assisted by OpenAI Codex (GPT-6); release date unavailable.
"""
import ast,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; PAPER=ROOT/'Paper'
Q3=ROOT/('问题三' if (ROOT/'问题三').is_dir() else '问题3')
SOURCES=[PAPER/'manuscript'/f'question{i}.md' for i in range(1,4)]
ORIGINALS=[ROOT/'问题1/问题1正文稿.md',PAPER/'references/q2-update/question2-source.md',Q3/'paper/问题三_不确定性融合论文.md']
TITLES=['问题一：三模态特征与多粒度时序对应','问题二：局部缺失下的情感预测','问题三：可解释融合与证据定位']
FORMAT='markdown+pipe_tables+tex_math_dollars+raw_tex+table_captions+autolink_bare_uris-smart'
def pandoc(text,from_format=FORMAT,to='json'):
 return subprocess.check_output(['pandoc','--from',from_format,'--to',to,'--wrap=none'],input=text.encode()).decode().replace('\r\n','\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw(t):return {'t':'RawBlock','c':['latex',t]}
def main():
 records=[];refs=[];keys={};assets={};used=[]
 tree=ast.parse((Q3/'paper/build_paper.py').read_text())
 formulas=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LATEX' for t in n.targets))
 for number,(path,title) in enumerate(zip(SOURCES,TITLES),1):
  text=path.read_text();parts=text.split('## 参考文献',1);text=parts[0];local={}
  for index,entry in re.findall(r'^\[(\d+)\]\s*(.+)$',parts[1] if len(parts)>1 else '',re.M):
   normalized='bert-2019' if 'BERT: Pre-training' in entry else re.sub(r'\W+','',entry.lower())
   if normalized not in keys:keys[normalized]=f'q{number}-ref{index}'
   local[index]=(keys[normalized],entry)
  if number==3:
   def eq(m):
    i=int(m[1]);return '$$\n'+r'\begin{gathered}'+r' \\ '.join(formulas[i-1])+r'\end{gathered}'+f'\\tag{{{i}}}'+'\n$$'
   text=re.sub(r'^[^\n#]+　（(\d+)）[ \t]*$',eq,text,flags=re.M)
  def cite(m):
   # Bracketed source positions are indices, even when the number is a reference key.
   if text[max(0,m.start()-2):m.start()]=='位置':return m[0]
   ids=re.split('[,，]',m[1])
   if not all(i in local for i in ids):return m[0]
   for i in ids:
    key,entry=local[i]
    if key not in used:used.append(key);refs.append((key,entry))
   return r'\cite{'+','.join(local[i][0] for i in ids)+'}'
  text=re.sub(r'(?<=[\w\u4e00-\u9fff])\[(\d+(?:[,，]\d+)*)\]',cite,text)
  text=text.replace('r=3 tanh(q)',r'$r=3\tanh(q)$').replace('ℓ′=ℓ+(0,−0.4,0)',r'$\ell^{\prime}=\ell+(0,-0.4,0)$').replace('ĉ=argmax ℓ′',r'$\hat c=\arg\max_c\ell^{\prime}_c$')
  text=text.replace(r'\\_',r'\_').replace('<br>','；').replace('∶',':')
  text=''.join(part if i%2 else re.sub(r'[-\w]+\$_\$\d+',lambda m:'`'+m[0]+'`',part) for i,part in enumerate(re.split(r'(`[^`\n]*`)',text)))
  # Resolve source identifiers to native LaTeX counters, including after pruning.
  table_ids=set(re.findall(r'^表(\d+)[^\n]*\n\n\|',text,re.M))
  figure_ids=set(re.findall(r'!\[图(\d+)',text))
  def caption(m):return ': '+m[2]+r'\label{q'+str(number)+'-tab'+m[1]+'}\n\n'
  text=re.sub(r'^表(\d+)\s*([^\n]*)\n\n(?=\|)',caption,text,flags=re.M)
  def figcap(m):return '!['+m[2]+r'\label{q'+str(number)+'-fig'+m[1]+'}]'
  text=re.sub(r'!\[图(\d+)\s*([^\]]*)\]',figcap,text)
  def tabref(m):
   assert m[2] in (table_ids if m[1]=='表' else figure_ids),(number,'dangling reference',m[0])
   return m[1]+r'\ref{q'+str(number)+('-tab' if m[1]=='表' else '-fig')+m[2]+'}'
  text=re.sub(r'(?<![词代])(表|图)(\d+)(?![\d])',tabref,text)
  text=re.sub(r'式（(\d+)）',lambda m:r'式\eqref{q'+str(number)+'-eq'+m[1]+'}',text)
  text=re.sub(r'\\tag\{(\d+)\}',lambda m:r'\label{q'+str(number)+'-eq'+m[1]+'}',text)
  doc=json.loads(pandoc(text));ni=0;nt=0;table_layouts=[]
  def transform(node):
   nonlocal ni
   if isinstance(node,dict):
    typ=node.get('t')
    if typ=='Code':
     escaped=pandoc(json.dumps({'pandoc-api-version':doc['pandoc-api-version'],'meta':{},'blocks':[{'t':'Plain','c':[{'t':'Str','c':node['c'][-1]}]}]}),'json','latex').strip()
     escaped=re.sub(r'([A-Za-z0-9]{8})',r'\1\\allowbreak{}',escaped)
     for a,b in [('-',r'-\allowbreak{}'),(r'\$',r'\$\allowbreak{}'),(r'\_',r'\_\allowbreak{}'),('/',r'/\allowbreak{}')]:escaped=escaped.replace(a,b)
     node.clear();node.update({'t':'RawInline','c':['latex',r'\texttt{'+escaped+'}']})
    elif typ=='Math' and node['c'][0]['t']=='DisplayMath':
     formula=node['c'][1];node.clear();node.update({'t':'RawInline','c':['latex',r'\begin{equation}'+formula+r'\end{equation}']})
    elif typ=='Header':node['c'][1][0]=f'q{number}-'+node['c'][1][0]
    elif typ=='Image':
     ni+=1;image_path=node['c'][2][0]
     if image_path.startswith('../问题三/'):image_path='../'+Q3.name+'/'+image_path[len('../问题三/'):]
     original=(PAPER/image_path).resolve();assert original.is_file(),original
     selected=original.with_suffix('.pdf') if original.with_suffix('.pdf').is_file() else original
     relative=selected.relative_to(ROOT).as_posix();node['c'][2][0]='../'+relative
     assets[relative]={'sha256':digest(selected),'source_image':original.relative_to(ROOT).as_posix()}
     if 'evidence_navigation/frames' in relative:node['c'][0][2].append(['width','55%'])
    for v in list(node.values()):transform(v)
   elif isinstance(node,list):
    for v in node:transform(v)
  blocks=[];heading='结果汇总'
  for block in doc['blocks']:
   if block['t']=='Header':heading=''.join(x.get('c','') for x in block['c'][2] if x['t']=='Str')
   if block['t']=='Table':
    nt+=1;cols=len(block['c'][2]);wide=cols>7
    if not block['c'][0]:block['c'][0]=[{'t':'Str','c':heading}]
    def size(node):
     if isinstance(node,dict):
      if node.get('t') in ('Str','Code','Math'):
       val=node['c'] if node['t']=='Str' else node['c'][-1];return sum(2 if ord(c)>255 else 1 for c in val)
      return sum(size(v) for v in node.values())
     return sum(size(v) for v in node) if isinstance(node,list) else 0
    rows=[block['c'][3]]+block['c'][4]
    weights=[max(10,min(44,max(size(row[i]) for row in rows))) for i in range(cols)]
    # Explicit widths preserve identifiers, uncertainty estimates and time ranges.
    layouts={
     (1,1):[24,35,41],(1,2):[26,44,30],(1,3):[15,85],
     (1,4):[40,30,30],(1,5):[40,14,19,10,17],(1,6):[30,12,58],
     (1,7):[30,12,15,8,8,27],(1,8):[33,12,12,26,17],
     (2,1):[23,44,33],(2,2):[8,92],(2,3):[31,69],
     (2,4):[35,13,14,13,14,11],(2,5):[24,15,16,15,16,14],
     (2,6):[29,28,28,15],(2,7):[29,17,16,38],
     (2,8):[25,25,25,25],(2,9):[13,12,21,18,18,18],
     (2,10):[28,36,36],(2,11):[30,20,25,25],
     (3,1):[44,56],(3,2):[34,16,18,16,16],(3,3):[34,16,18,16,16],
     (3,4):[31,20,49],(3,5):[22,22,20,20,16],(3,6):[34,20,20,26],
     (3,7):[32,15,19,17,17],(3,8):[43,19,19,19],(3,9):[46,18,18,18],
     (3,10):[8,13,18,14,14,14,19],(3,11):[7,34,16,28,15],
     (3,12):[34,22,22,22],
    }
    weights=layouts.get((number,nt),weights)
    fractions=[w/sum(weights) for w in weights]
    block['c'][2]=fractions
    # Numeric columns align by their right edge; identifiers and prose stay left.
    def plain(node):
     if isinstance(node,dict):
      if node.get('t')=='Str':return node['c']
      if node.get('t')=='Space':return ' '
      return ''.join(plain(v) for v in node.values())
     if isinstance(node,list):return ''.join(plain(v) for v in node)
     return ''
    numeric=[]
    for i in range(cols):
     values=[plain(row[i]) for row in block['c'][4]]
     numeric.append(all(re.fullmatch(r'[+−\-\d.,/%()（）± eE—]+',v) for v in values))
    table_layouts.append((fractions,numeric,len(block['c'][4])))
    blocks.extend([raw(r'\begin{PaperWideTable}' if wide else r'\begin{PaperTable}{'+str(cols)+'}'),block,raw(r'\end{PaperWideTable}' if wide else r'\end{PaperTable}')])
   else:blocks.append(block)
  doc['blocks']=blocks;transform(doc)
  target=PAPER/'chapters'/f'question{number}.tex'
  latex=pandoc(json.dumps(doc,ensure_ascii=False),'json','latex')
  layout_iter=iter(table_layouts)
  def continued(m):
   table=m[0];fractions,numeric,row_count=next(layout_iter);col_count=len(fractions);cell_index=0
   def cell(match):
    nonlocal cell_index
    col=cell_index%col_count;cell_index+=1
    # Pandoc 2.9 rounds relative widths; set exact widths after subtracting gutters.
    width=r'\dimexpr '+f'{fractions[col]:.6f}'+r'\linewidth-'+f'{fractions[col]*2*(col_count-1):.6f}'+r'\tabcolsep\relax'
    alignment=r'\raggedleft' if numeric[col] else r'\raggedright'
    return r'\begin{minipage}['+match[1]+']{'+width+'}'+alignment
   table=re.sub(r'\\begin\{minipage\}\[([bt])\]\{[0-9.]+\\columnwidth\}\\(?:raggedright|raggedleft|centering)',cell,table)
   caption=re.search(r'\\caption\{(.*?)\}\\tabularnewline',table,re.S)
   if row_count<=20 and caption:
    # Keep short result tables intact. Long sample tables retain repeated headers.
    spec=re.search(r'\\begin\{longtable\}\[\]\{(.*?)\}\n',table)[1]
    header=table.split(r'\toprule',1)[1].split(r'\endfirsthead',1)[0]
    body=table.split(r'\endhead',1)[1].split(r'\end{longtable}',1)[0]
    return (r'\begin{table}[!htbp]'+ '\n'+r'\centering\caption{'+caption[1]+'}\n'+
            r'\begin{tabular}{'+spec+'}\n'+r'\toprule'+header+body+r'\end{tabular}'+ '\n'+r'\end{table}')
   if caption:
    title=re.sub(r'\\label\{[^}]+\}','',caption[1])
    table=table.replace(r'\endfirsthead',r'\endfirsthead'+'\n'+r'\caption[]{'+title+'（续）}'+r'\tabularnewline',1)
   return table
  latex=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',continued,latex,flags=re.S)
  # Adjacent media examples share one row and retain separate figure counters.
  def paired_frames(match):
   parts=[]
   for body in (match[1],match[2]):
    if 'evidence_navigation/frames/' not in body:return match[0]
    body=re.sub(r'\\includegraphics\[[^\]]*\]',lambda _:r'\includegraphics[width=\linewidth,height=.22\textheight,keepaspectratio]',body)
    parts.append(r'\begin{minipage}[t]{.48\linewidth}'+body+r'\end{minipage}')
   return r'\begin{figure}[H]'+'\n'+r'\centering'+parts[0]+r'\hfill'+parts[1]+'\n'+r'\end{figure}'
  latex=re.sub(r'\\begin\{figure\}\s*((?:(?!\\end\{figure\}).)*)\\end\{figure\}\s*\\begin\{figure\}\s*((?:(?!\\end\{figure\}).)*)\\end\{figure\}',paired_frames,latex,flags=re.S)
  target.write_text('% Generated from Paper/manuscript; edit Markdown source.\n'+f'\\section{{{title}}}\n'+latex)
  records.append({'question':number,'source':path.relative_to(ROOT).as_posix(),'source_sha256':digest(path),'research_source':ORIGINALS[number-1].relative_to(ROOT).as_posix(),'research_sha256':digest(ORIGINALS[number-1]),'chapter_sha256':digest(target),'images':ni,'tables':nt})
 bibliography=[r'\begin{thebibliography}{99}',r'\raggedright']
 for key,entry in refs:
  bibliography.extend([r'\bibitem{'+key+'}',pandoc(re.sub(r'(https?://[^\s，。<>]+)',r'<\1>',entry),FORMAT,'latex').strip()])
 bibliography.append(r'\end{thebibliography}');(PAPER/'chapters/references.tex').write_text('\n'.join(bibliography)+'\n')
 inputs=['main.tex','preamble.tex','abstract.tex','conclusion.tex','introduction.tex','data.tex','chapters/references.tex','assets/official-cover.pdf','assets/figure_sources.json','assets/q2_confusion.svg','assets/q2_missing.svg']
 (PAPER/'sources.json').write_text(json.dumps({'sources':records,'assets':assets,'references':len(refs),'citation_order':used,'paper_inputs':{n:digest(PAPER/n) for n in inputs},'q3_equation_generator_sha256':digest(Q3/'paper/build_paper.py'),'generator_sha256':digest(Path(__file__))},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'chapters':3,'figures':sum(r['images'] for r in records),'tables':sum(r['tables'] for r in records),'references':len(refs)}))
if __name__=='__main__':main()
