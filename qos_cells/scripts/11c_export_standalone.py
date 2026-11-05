"""Export the Slides deck (project/slides/*.html) to one offline HTML file and a PDF.
Images and fonts are embedded as base64, so the file works without internet."""
import base64, json, re, sys
from playwright.sync_api import sync_playwright
ROOT="/home/claude/final"; OUT=sys.argv[1] if len(sys.argv)>1 else f"{ROOT}/out"
import os; os.makedirs(OUT,exist_ok=True)
deck=json.load(open(f"{ROOT}/deck/project/deck.json"))
blob={"bb0b2e3f4036a492ca776a5743f3c55a":"classes","a5ed6e55ad4799332861ffefb06fe0e6":"rarecost","b86ae1f736d1503ac68a94b8888c4d7a":"hardware","a196ddd718cf93c646bbcebf624725b2":"memory"}
b64=lambda p: base64.b64encode(open(p,"rb").read()).decode()
F=f"{ROOT}/fonts/node_modules/@fontsource"
faces="".join(f"@font-face{{font-family:'{fam}';font-weight:{w};src:url(data:font/woff2;base64,{b64(f'{F}/{pkg}/files/{pkg}-latin-{w}-normal.woff2')}) format('woff2')}}"
  for fam,pkg,ws in [("Space Grotesk","space-grotesk",[400,700]),("Inter","inter",[400,600,700])] for w in ws)
ARROW='<svg width="56" height="28" viewBox="0 0 56 28" style="flex:none"><path d="M0 9h34V0l22 14-22 14v-9H0z" fill="#08090B"/></svg>'
slides=[]; notes=[]
for sid in deck["order"]:
    h=open(f"{ROOT}/deck/project/slides/{sid}.html").read()
    for k,v in blob.items(): h=h.replace(f"/_blob/{k}",f"data:image/png;base64,{b64(f'{ROOT}/fig/{v}.png')}")
    h=re.sub(r'<x-shape kind="arrow-right"[^>]*></x-shape>',ARROW,h)
    n=re.search(r"<aside>(.*?)</aside>",h,re.S); notes.append(n.group(1) if n else "")
    slides.append(re.sub(r"<aside>.*?</aside>","",h,flags=re.S))
css=faces+"""
*{margin:0;padding:0;box-sizing:border-box}
html,body{background:#2b2b2b;height:100%;overflow:hidden}
#stage{position:fixed;left:50%;top:50%;width:1920px;height:1080px;transform-origin:center center}
section{position:absolute;inset:0;width:1920px;height:1080px;display:none!important;overflow:hidden}
section.on{display:flex!important}
table{border-collapse:collapse;width:100%} td,th{text-align:left;padding:.35em .6em;border-bottom:1px solid #E2DFD8;vertical-align:top} th{font-weight:600}
#hint{position:fixed;bottom:8px;left:50%;transform:translateX(-50%);font:13px Inter,sans-serif;color:#aaa}
@media print{@page{size:1920px 1080px;margin:0}html,body{background:#fff;overflow:visible;height:auto}#stage{position:static;transform:none!important;width:1920px;height:auto}
section{position:relative!important;display:flex!important;page-break-after:always;break-after:page}#hint{display:none}}
"""
js="""
const S=[...document.querySelectorAll('section')];let i=Math.max(0,(parseInt(location.hash.slice(1))||1)-1);
function show(k){i=Math.max(0,Math.min(S.length-1,k));S.forEach((s,j)=>s.classList.toggle('on',j===i));history.replaceState(null,'','#'+(i+1));}
function fit(){const st=document.getElementById('stage');const sc=Math.min(innerWidth/1920,innerHeight/1080);st.style.transform=`translate(-50%,-50%) scale(${sc})`;}
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(i+1)}if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(i-1)}if(e.key==='Home')show(0);if(e.key==='End')show(S.length-1);if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
addEventListener('click',e=>show(e.clientX>innerWidth/2?i+1:i-1));addEventListener('resize',fit);fit();show(i);
"""
notes_html="".join(f"<!-- slide {k+1} notes: {n.replace('--','–')} -->\n" for k,n in enumerate(notes))
html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{deck['title']}</title><style>{css}</style></head><body>
{notes_html}<div id="stage">{''.join(slides)}</div><div id="hint">← → or click · F fullscreen · Ctrl+P for PDF</div><script>{js}</script></body></html>"""
open(f"{OUT}/final_slides.html","w").write(html)
with sync_playwright() as p:
    b=p.chromium.launch(executable_path="/usr/bin/google-chrome"); pg=b.new_page(viewport={"width":1920,"height":1080})
    pg.goto(f"file://{OUT}/final_slides.html"); pg.wait_for_timeout(800)
    for k in range(len(slides)):
        pg.goto(f"file://{OUT}/final_slides.html#{k+1}"); pg.reload(); pg.wait_for_timeout(300); pg.screenshot(path=f"{OUT}/prev_{k+1}.png")
    pg.emulate_media(media="print"); pg.pdf(path=f"{OUT}/final_slides.pdf",width="1920px",height="1080px",print_background=True)
    b.close()
print("ok", os.path.getsize(f"{OUT}/final_slides.html"), os.path.getsize(f"{OUT}/final_slides.pdf"))
