"""Render edited manuscripts; never import over the curated Markdown sources.
Build-script editing assisted by OpenAI Codex (GPT-6); release date unavailable.
"""
import ast,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; PAPER=ROOT/'Paper'
SOURCES=[PAPER/'manuscript'/f'question{i}.md' for i in range(1,4)]
ORIGINALS=[ROOT/'问题1/问题1正文稿.md',ROOT/'问题2/docs/paper_current/reports/E题问题2统一论文素材与当前模型结果.md',ROOT/'问题三/paper/问题三_不确定性融合论文.md']
TITLES=['问题一：三模态特征与多粒度时序对应','问题二：局部缺失下的情感预测','问题三：可解释融合与证据定位']
FORMAT='markdown+pipe_tables+tex_math_dollars+raw_tex+table_captions+autolink_bare_uris-smart'
def pandoc(text,from_format=FORMAT,to='json'):
 return subprocess.check_output(['pandoc','--from',from_format,'--to',to,'--wrap=none'],input=text.encode()).decode()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw(t):return {'t':'RawBlock','c':['latex',t]}
def main():
 records=[];refs=[];keys={};assets={};used=[]
 tree=ast.parse((ROOT/'问题三/paper/build_paper.py').read_text())
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
  doc=json.loads(pandoc(text));ni=0;nt=0
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
     ni+=1;original=(PAPER/node['c'][2][0]).resolve();assert original.is_file(),original
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
    if number==1 and len(block['c'][4])==100:weights=[36,12,12,29,16]
    if number==3 and len(block['c'][4])==20 and cols==7:weights=[6,10,15,10,10,10,15]
    block['c'][2]=[w/sum(weights) for w in weights]
    blocks.extend([raw(r'\begin{PaperWideTable}' if wide else r'\begin{PaperTable}{'+str(cols)+'}'),block,raw(r'\end{PaperWideTable}' if wide else r'\end{PaperTable}')])
   else:blocks.append(block)
  doc['blocks']=blocks;transform(doc)
  target=PAPER/'chapters'/f'question{number}.tex'
  latex=pandoc(json.dumps(doc,ensure_ascii=False),'json','latex')
  def continued(m):
   table=m[0];caption=re.search(r'\\caption\{(.*?)\}\\tabularnewline',table,re.S)
   if caption:
    title=re.sub(r'\\label\{[^}]+\}','',caption[1])
    table=table.replace(r'\endfirsthead',r'\endfirsthead'+'\n'+r'\caption[]{'+title+'（续）}'+r'\tabularnewline',1)
   return table
  latex=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',continued,latex,flags=re.S)
  target.write_text('% Generated from Paper/manuscript; edit Markdown source.\n'+f'\\section{{{title}}}\n'+latex)
  records.append({'question':number,'source':path.relative_to(ROOT).as_posix(),'source_sha256':digest(path),'research_source':ORIGINALS[number-1].relative_to(ROOT).as_posix(),'research_sha256':digest(ORIGINALS[number-1]),'chapter_sha256':digest(target),'images':ni,'tables':nt})
 bibliography=[r'\begin{thebibliography}{99}',r'\raggedright']
 for key,entry in refs:
  bibliography.extend([r'\bibitem{'+key+'}',pandoc(re.sub(r'(https?://[^\s，。<>]+)',r'<\1>',entry),FORMAT,'latex').strip()])
 bibliography.append(r'\end{thebibliography}');(PAPER/'chapters/references.tex').write_text('\n'.join(bibliography)+'\n')
 inputs=['main.tex','preamble.tex','abstract.tex','conclusion.tex','introduction.tex','data.tex','chapters/references.tex','assets/official-cover.pdf','assets/figure_sources.json','assets/q2_confusion.svg','assets/q2_missing.svg']
 (PAPER/'sources.json').write_text(json.dumps({'sources':records,'assets':assets,'references':len(refs),'citation_order':used,'paper_inputs':{n:digest(PAPER/n) for n in inputs},'q3_equation_generator_sha256':digest(ROOT/'问题三/paper/build_paper.py'),'generator_sha256':digest(Path(__file__))},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'chapters':3,'figures':sum(r['images'] for r in records),'tables':sum(r['tables'] for r in records),'references':len(refs)}))
if __name__=='__main__':main()
