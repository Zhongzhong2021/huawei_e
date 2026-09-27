"""Convert the official DOC cover, retaining artwork and removing its page-0 footer.
Windows uses Microsoft Word; other platforms use LibreOffice and Poppler.
The source Word document is never saved or overwritten.
Code edited with OpenAI Codex (GPT-6); release date unavailable.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as E
import subprocess,shutil,os
P=Path(__file__).resolve().parents[1]
if os.name == 'nt':
 with TemporaryDirectory(prefix='paper-cover-') as folder:
  result=Path(folder)/'page-1.pdf'
  subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(P/'scripts/export_cover_word.ps1'),'-SourcePath',str(P/'references/official/template-2026.doc'),'-OutputPath',str(result)],check=True,timeout=120)
  if not result.is_file() or not result.read_bytes().startswith(b'%PDF-'):
   raise RuntimeError('Word cover conversion did not produce a PDF.')
  shutil.copyfile(result,P/'assets/official-cover.pdf')
 print('Cover updated from saved template-2026.doc.')
 raise SystemExit(0)
with TemporaryDirectory(prefix='paper-cover-') as folder:
 d=Path(folder);profile='-env:UserInstallation='+str((d/'profile').as_uri())
 def convert(p,ext):subprocess.run(['libreoffice',profile,'--headless','--convert-to',ext,'--outdir',str(d),str(p)],check=True)
 convert(P/'references/official/template-2026.doc','docx')
 with ZipFile(d/'template-2026.docx') as z,ZipFile(d/'cover.docx','w',ZIP_DEFLATED) as out:
  for n in z.namelist():
   b=z.read(n)
   if n.startswith('word/footer') and n.endswith('.xml'):
    root=E.fromstring(b)
    for node in list(root):root.remove(node)
    E.SubElement(root,'{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')
    b=E.tostring(root,encoding='utf-8',xml_declaration=True)
   out.writestr(n,b)
 convert(d/'cover.docx','pdf')
 subprocess.run(['pdfseparate','-f','1','-l','1',str(d/'cover.pdf'),str(d/'page-%d.pdf')],check=True)
 shutil.copyfile(d/'page-1.pdf',P/'assets/official-cover.pdf')
