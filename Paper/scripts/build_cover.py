"""Convert the official DOC cover, retaining artwork and removing its page-0 footer.
Requires LibreOffice and Poppler. Normal builds use the committed PDF asset.
Code edited with OpenAI Codex (GPT-6); release date unavailable.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as E
import subprocess,shutil
P=Path(__file__).resolve().parents[1]
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
