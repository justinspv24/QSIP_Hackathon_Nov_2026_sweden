import json, datetime
EQ=json.load(open("/home/claude/final/eq/eqs_out.json"))
def eq(k): return EQ[k]["svg"]
IMG={"classes":"/_blob/bb0b2e3f4036a492ca776a5743f3c55a","rarecost":"/_blob/a5ed6e55ad4799332861ffefb06fe0e6",
     "hardware":"/_blob/b86ae1f736d1503ac68a94b8888c4d7a","memory":"/_blob/a196ddd718cf93c646bbcebf624725b2"}
BG="#FFFFFF"; BG2="#F5F3EF"; INK="#08090B"; BODY="#4A4A48"; YEL="#FFC900"; BLUE=INK; ORNG=INK; LINE="#E2DFD8"
HEAD="'Space Grotesk', Arial, sans-serif"; TEXT="'Inter', Arial, sans-serif"
N=7
def sec(id,i,inner,notes,bg=BG,extra=""):
    foot=(f'<p style="position:absolute;left:128px;bottom:64px;width:1100px;font-size:24px;color:#6B6A66">Streaming rare cells into qubits · QSIP Hackathon 2026</p>'
          f'<p style="position:absolute;right:128px;bottom:64px;width:200px;font-size:24px;color:#6B6A66;text-align:right">{i} / {N}</p>')
    return (f'<section id="{id}" data-transition="fade" style="background:{bg};color:{INK};font-family:{TEXT};padding:112px 128px 160px;'
            f'display:flex;flex-direction:column;gap:36px{extra}">\n{inner}\n{foot}\n<aside>{notes}</aside>\n</section>\n')
def eyebrow(t): return f'<div style="display:flex;align-items:center;gap:16px"><div style="width:44px;height:10px;background:{YEL};border-radius:2px"></div><p style="font-family:{HEAD};font-size:24px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:{INK}">{t}</p></div>'
def h2(t): return f'<h2 style="font-family:{HEAD};font-size:60px;font-weight:700;line-height:1.1;color:{INK}">{t}</h2>'
def head(e,t): return f'<div style="display:flex;flex-direction:column;gap:16px">{eyebrow(e)}{h2(t)}</div>'
def img(k,w,ratio,alt): return f'<img src="{IMG[k]}" alt="{alt}" style="width:{w}px;height:{round(w/ratio)}px;object-fit:contain">'
def card(inner,bg="#FFFFFF",border=LINE,pad=36,gap=16,flex="1"):
    return f'<div style="flex:{flex};display:flex;flex-direction:column;gap:{gap}px;background:{bg};border:1px solid {border};border-radius:20px;padding:{pad}px">{inner}</div>'
def p(t,size=30,color=BODY,extra=""): return f'<p style="font-size:{size}px;line-height:1.35;color:{color}{extra}">{t}</p>'
def h3(t,color=INK,size=36): return f'<h3 style="font-family:{HEAD};font-size:{size}px;font-weight:700;line-height:1.15;color:{color}">{t}</h3>'
def big(n,label,color=INK,bar=None):
    return (f'<div style="display:flex;flex-direction:column;gap:4px;border-left:{8 if bar else 4}px solid {bar or color};padding:4px 0 4px 28px">'
            f'<p style="font-family:{HEAD};font-size:64px;font-weight:700;line-height:1.05;color:{color}">{n}</p>{p(label,28)}</div>')
def eqbox(k): return f'<div style="display:flex;justify-content:center;align-items:center">{eq(k)}</div>'
slides=[]

# 1 cover
steps=["Dataset","Challenge","Quantum idea","Prototype","Real world"]
pills="".join(f'<div style="display:flex;align-items:center;gap:14px"><p style="font-family:{HEAD};font-size:24px;font-weight:700;color:{INK};background:{YEL};padding:6px 14px;border-radius:999px">{i+1}</p>{p(s,28,INK)}</div>'
              +('<p style="font-size:28px;color:#A9AEBB">→</p>' if i<4 else "") for i,s in enumerate(steps))
slides.append(("cover",sec("cover",1,
 f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:40px">'
 f'{eyebrow("QSIP Hackathon · ExtremeData4QML · Stockholm · 6 Nov 2026")}'
 f'<h1 style="font-family:{HEAD};font-size:96px;font-weight:700;line-height:1.05;color:{INK}">Learning rare cells from a<br>data stream too big to store</h1>'
 f'<div style="width:420px;height:14px;background:{YEL};border-radius:3px"></div>'
 f'{p("Quantum oracle sketching: every record becomes one quantum gate, and is then forgotten.",36)}</div>'
 f'<div style="display:flex;align-items:center;gap:22px;border-top:1px solid {LINE};padding:36px 0 0 0">{pills}</div>',
 "Hook. One in 70 blood cells is a dendritic cell; one in a million is a tumour cell. Cell atlases hold around 100 million cells, so you cannot store the stream. Question: can a small quantum register learn the rare class straight from the stream? We follow the five challenge steps on one dataset.")))

# 2 data
slides.append(("data",sec("data",2,
 head("Step 1 · Dataset  ·  Step 2 · Challenge","Blood cells: mostly zeros, and the key ones are rare")+
 f'<div style="flex:1;display:flex;gap:64px;align-items:center">'
 f'<div style="display:flex;flex-direction:column;gap:12px">{img("classes",900,1.765,"Bar chart of cells per type in PBMC3k on a log scale; dendritic cells 37 (1.4 %) and megakaryocytes 15 (0.6 %) are highlighted as rare")}'
 f'{p("PBMC3k: 2,638 blood cells, 10x Genomics",24,"#6B6A66")}</div>'
 f'<div style="flex:1;display:flex;flex-direction:column;gap:40px">'
 f'{big("32,738","genes measured per cell")}{big("94 %","of the values are zero")}{big("1 in 71","is a dendritic cell; 1 in 176 a megakaryocyte",INK,YEL)}'
 f'</div></div>',
 "PBMC3k: 32,738 genes measured, about 94 percent zeros among the expressed genes, 2,638 cells, and two classes under 2 percent. The hard part is not accuracy on 2,638 cells; classical solves that. It is the combination: memory grows with the number of features, the data arrives as a one-pass stream, and the class you need is about 1 percent of it. Real atlases hold 10^7 to 10^8 cells.")))

# 3 technique
def stepcard(n,t,s):
    return card(f'<p style="font-family:{HEAD};font-size:24px;font-weight:700;color:{INK};background:{YEL};padding:2px 12px;border-radius:999px;align-self:flex-start">{n}</p>{h3(t,size=34)}{p(s,26)}',pad=32,gap=10)
arrow=f'<x-shape kind="arrow-right" style="background:{INK};width:56px;height:28px"></x-shape>'
slides.append(("technique",sec("technique",3,
 head("Step 3 · The quantum idea","Each record nudges a phase, then it is gone")+
 f'<div style="display:flex;align-items:center;gap:20px">'
 f'{stepcard("01","Stream of records","(cell, gene, value, label): read once, never stored, zeros skipped")}{arrow}'
 f'{stepcard("02","One phase gate each","All gates commute, so the register keeps only their sum")}{arrow}'
 f'{stepcard("03","One measurement","A Hadamard test reads out a score: rare or common")}</div>'
 f'<div style="display:flex;justify-content:space-between;align-items:center;background:{BG2};border-radius:20px;padding:28px 64px">'
 f'{eqbox("qubits")}<div style="width:2px;height:120px;background:#DDE1E8"></div>{eqbox("phase")}<div style="width:2px;height:120px;background:#DDE1E8"></div>{eqbox("score")}</div>'
 +p("Like rain on drums: no drum keeps a single drop, but each ends up tuned by how much fell on it.",30,INK,";font-style:italic"),
 "Mechanism from Zhao, Zlokapa, Preskill and Huang 2026. Each streamed record is one diagonal phase gate on the basis state of its gene. The gates commute, so after M records the phase on gene g is the weighted sum of its kicks. A Hadamard test then reads the sum of squared amplitudes times sine of the phase, which is a linear classifier score. N features need only log2 N qubits. No QRAM: the sketching is the data loading.")))

# 4 why it fits
slides.append(("fits",sec("fits",4,
 head("Step 3 · Why it fits this data","Memory stays tiny; rare classes cost samples, not qubits")+
 f'<div style="flex:1;display:flex;gap:56px;align-items:center">'
 f'<div style="flex:1;display:flex;flex-direction:column;gap:28px">'
 +card(f'{h3("Memory: few qubits; price: more records",size=32)}{eqbox("err")}{p("Error falls as records stream in; measured with no fitted parameter.",26)}',pad=32,gap=12)
 +card(f'{h3("Our addition: fair to rare classes",BLUE,32)}{eqbox("rare")}{p("Weighted gates keep rare classes fair; they pay in records: dendritic cells need ×72.",26)}',pad=32,gap=12,bg="#FFF8DB",border=YEL)
 +f'</div><div style="display:flex;flex-direction:column;gap:12px">{img("rarecost",860,1.608,"Bar chart: extra samples needed versus class rarity; prediction ×2.8, ×13, ×50 and measured ×3.1, ×13, ×71 for class shares 0.1, 0.02, 0.005")}'
 f'{p("Synthetic test: same data, only the rare share changes",24,"#6B6A66")}</div></div>',
 "Each claim comes with a measurement. Memory is log2 N qubits; the price is samples, error squared about C times N over M, and the closed-form prediction tracks the measurement with no fitted parameter. Our twist: plus or minus one over pi weights keep the mean unbiased but square inside the variance, giving a sample cost of 1 over pi times 1 minus pi. Measured ×71 versus law ×50 at pi = 0.005. Dendritic cells pay ×72 over a balanced class.")))

# 5 prototype
def td(t,bold=False,color=INK): return f'<td style="font-size:28px;color:{color}{";font-weight:700" if bold else ""}">{t}</td>'
tbl=(f'<table style="font-family:{TEXT};font-size:28px;color:{INK}">'
 f'<tr><th style="width:34%;font-size:24px;color:{BODY}">device</th><th style="width:33%;font-size:24px;color:{BODY}">3 qubits</th><th style="width:33%;font-size:24px;color:{BODY}">5 qubits</th></tr>'
 f'<tr>{td("IQM Garnet",True)}{td("8 / 8 right")}{td("5 / 8 → 8 / 8")}</tr>'
 f'<tr>{td("signal kept",color=BODY)}{td("78 %",color=BODY)}{td("43 %",color=BODY)}</tr>'
 f'<tr>{td("IBM Kingston",True)}{td("8 / 8 right")}{td("8 / 8 right")}</tr>'
 f'<tr>{td("signal kept",color=BODY)}{td("89 %",color=BODY)}{td("74 %",color=BODY)}</tr></table>')
slides.append(("prototype",sec("prototype",5,
 head("Step 4 · Prototype","It runs on real quantum hardware: 8 of 8 test cells right")+
 f'<div style="flex:1;display:flex;gap:48px;align-items:center">'
 f'<div style="display:flex;flex-direction:column;gap:12px">{img("hardware",900,1.827,"Bar chart of classifier scores for 8 held-out cells: exact formula, quantum simulator and real IQM Garnet agree in sign for all 8 cells")}'
 f'{p("Same 8 held-out cells, same frozen model · 2,000 shots",24,"#6B6A66")}</div>'
 f'<div style="flex:1;display:flex;flex-direction:column;gap:28px">{tbl}'
 +card(f'{p("Noise shrinks the score. Five known circuits measure the shrink and undo it:",26,BODY)}{eqbox("cal")}{p("Trained in simulation; the hardware runs inference only.",26,INK)}',pad=28,gap=10,bg=BG2,border=BG2)
 +'</div></div>',
 "Formula equals statevector to 1e-15; the shot simulator is within 0.03. IQM Garnet, 3 qubits, 11 CZ: all 8 cells right before and after calibration, signal kept 78 percent. At 16 genes, 53 CZ, raw signs break (5 of 8) and calibration rescues them (8 of 8). Same circuits on IBM Kingston: 8 of 8 raw at both sizes, signal kept 89 and 74 percent, though heavy-hex routing needed 64 two-qubit gates. One run per device, about 7 hours apart. Phases are frozen and hashed; no training on hardware.")))

# 6 honest check
slides.append(("check",sec("check",6,
 head("Honest check · could classical do it with little memory?","Yes: on real data, a small classical sketch is enough")+
 f'<div style="flex:1;display:flex;gap:56px;align-items:center">'
 f'<div style="display:flex;flex-direction:column;gap:12px">{img("memory",940,1.895,"Bar chart of memory needed: blood-cell gene pairs 2.1 M features but 256 counters suffice; genomes 791 k features, 16 k counters; synthetic no-structure data needs 262 k of 2.1 M")}'
 f'{p("Memory needed to match the full classifier · 5 seeds per point",24,"#6B6A66")}</div>'
 f'<div style="flex:1;display:flex;flex-direction:column;gap:36px">'
 f'<div style="display:flex;flex-direction:column;gap:8px">{h3("Blood cells",size=34)}{p("The signal sits in a few marker genes: 256 numbers do the job.",28)}</div>'
 f'<div style="display:flex;flex-direction:column;gap:8px">{h3("Human genomes",size=34)}{p("Signal spread over many SNPs, but compressible: memory stays flat as data grows.",28)}</div>'
 f'<div style="display:flex;flex-direction:column;gap:8px;border-left:8px solid {YEL};padding-left:24px">{h3("Only structureless data",INK,34)}{p("forces classical memory to grow with the data. That is where the quantum bound lives.",28)}</div>'
 f'</div></div>',
 "Route 2. We gave classical streaming sketches (count-sketch, feature hashing, sparse projections, heavy hitters, reservoir) the same records. Blood-cell gene pairs, N = 2.1 million: about 256 counters match the exact centroid, versus 22 index qubits for the quantum register. 1000 Genomes, 791 thousand SNPs, Europe and East Asia: 8 to 16 thousand counters, and memory barely grows with N (log-log slopes from -0.68 to 0.40). Only a synthetic control with no structure needs 12 percent of N or more. So no memory advantage on these datasets: the paper's lower bounds need signals that are dense and incompressible.")))

# 7 real world + verdict
def row(a,b,c,bold=False): return f'<tr>{td(a,bold)}{td(b,bold)}{td(c,bold)}</tr>'
cost=(f'<table style="font-family:{TEXT};font-size:28px;color:{INK}">'
 f'<tr><th style="width:36%;font-size:24px;color:{BODY}">one rare-cell sketch</th><th style="width:30%;font-size:24px;color:{BODY}">all genes</th><th style="width:34%;font-size:24px;color:{BODY}">all gene pairs</th></tr>'
 +row("quantum memory","30 logical qubits","60 logical qubits")+row("quantum time","≈ 11 minutes","≈ 1.5 years")
 +row("classical memory","131 kB","4.3 GB")+'</table>')
slides.append(("realworld",sec("realworld",7,
 head("Step 5 · Real world","Where this matters, and what it would really cost")+
 f'<div style="display:flex;gap:28px">'
 +card(f'{h3("Cell atlases",size=32)}{p("100 million cells, arriving as a stream",26)}',pad=28,gap=8)
 +card(f'{h3("Liquid biopsies",size=32)}{p("A tumour cell is 1 in a million blood cells",26)}',pad=28,gap=8)
 +card(f'{h3("Fraud and money laundering",size=32)}{p("0.1–0.2 % of transactions, endless streams",26)}',pad=28,gap=8)
 +'</div>'
 f'<div style="flex:1;display:flex;gap:48px;align-items:center">'
 f'<div style="flex:1.4;display:flex;flex-direction:column;gap:10px">{cost}{p("Fault-tolerant gates at 1 MHz, 10 % error, dendritic-cell rarity",24,"#6B6A66")}</div>'
 f'<div style="flex:1;display:flex;flex-direction:column;gap:14px;background:{INK};border-radius:20px;padding:40px">'
 f'{h3("What we show",color=YEL,size=36)}'
 f'<p style="font-size:28px;line-height:1.35;color:#F5F3EF">A working mechanism with measured costs, running on two real QPUs. Not a quantum advantage: that needs data that is dense <b>and</b> incompressible.</p></div>'
 f'</div>',
 "Ledger: all genes is 15 index qubits (about 30 logical), about 700 seconds per sketch, versus 131 kB classical, so classical wins easily. Gene pairs: 30 index qubits (about 60 logical) and about 1.5 years per sketch versus 4.3 GB classical. The gain is size, not time, and it needs fault tolerance. We do not claim that our centroid task inherits the paper's lower bound; Route 2 shows real biology is compressible. Next: a task whose decision direction is dense over a feature space far bigger than memory, and a classical-shadow readout.")))

import os
root="/home/claude/final/deck/project"
for id,html in slides: open(f"{root}/slides/{id}.html","w").write(html)
deck={"v":4,"createdOnFiles":{"v":1,"at":datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},"lists":"css",
 "title":"Final Slides – Streaming Rare Cells into Qubits","cover":"cover","order":[s[0] for s in slides],
 "sections":{"s1":{"description":"Hook: rare cells in a stream too big to store","start":"cover"},
   "s2":{"description":"Steps 1–2: the dataset and why it is hard","start":"data"},
   "s3":{"description":"Step 3: oracle sketching and why it fits","start":"technique"},
   "s4":{"description":"Step 4: prototype on simulator and two real QPUs","start":"prototype"},
   "s5":{"description":"Honest classical check and real-world cost","start":"check"}},
 "faces":{"space-grotesk":{"family":"Space Grotesk","href":"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400..700&display=swap"},
          "inter":{"family":"Inter","href":"https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"}},"designSystems":[]}
json.dump(deck,open(f"{root}/deck.json","w"),ensure_ascii=False,indent=1)
for id,_ in slides: print(id, os.path.getsize(f"{root}/slides/{id}.html"))
