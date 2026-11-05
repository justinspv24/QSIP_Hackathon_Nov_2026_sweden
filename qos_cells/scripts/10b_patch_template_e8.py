"""Insert the E8 backup slide (Route 2 on 1000 Genomes genotypes) into presentation/plan3_template.html (idempotent).
Everything is drawn from RESULTS.e8 (scripts/08_slides.py reads results/e8_route2_genotypes.json).
Run:  python scripts/10b_patch_template_e8.py && python scripts/08_slides.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import ROOT

TPL = ROOT / "presentation" / "plan3_template.html"
MARK = "<!-- B5 · E8: DENSE SIGNAL, 1000 GENOMES -->"

HTML = MARK + """
<section class="slide">
  <div class="step backup"><span class="num">B</span>Backup · a dense-signal dataset: 1000 Genomes (E8)</div>
  <h2 id="e8-title" style="font-size:29px"></h2>
  <div class="row">
    <div class="col" style="flex:1.05"><svg id="e8plot" viewBox="0 0 560 340" width="100%"></svg></div>
    <div class="col" style="gap:8px;justify-content:flex-start">
      <style>#e8table th,#e8table td{padding:6px 8px}</style>
      <table id="e8table" style="font-size:13px"></table>
      <p class="note" id="e8-note"></p>
    </div>
  </div>
</section>

"""

JS = r"""/* ---- B5: E8 genotypes ---- */
(function(){
  const e=R.e8; if(!e) return;
  const sup=n=>String(n).replace(/-/g,"⁻").replace(/\d/g,d=>"⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  const fmtR=v=>{if(v>=0.1)return v.toFixed(2);const ex=Math.floor(Math.log10(v));const m=Math.round(v/10**ex);return (m>=10?"1·10"+sup(ex+1):m+"·10"+sup(ex));};
  const fmtS=v=>Number.isInteger(Math.log2(v))?"2"+sup(Math.log2(v)):v.toLocaleString();
  const cols=[C.ink,C.gold,C.red];
  /* chart: best one-pass S* vs N, log-log */
  const svg=document.getElementById("e8plot");const W=560,H=340,L=62,T=26,B=58,pw=W-L-24,ph=H-T-B;
  const allN=e.tasks.flatMap(t=>t.rows.map(r=>r.N));const n0=Math.min(...allN),n1=Math.max(...allN);
  const lx0=Math.log2(n0)-0.3,lx1=Math.log2(n1)+0.3,ly0=6,ly1=Math.ceil(Math.log2(n1))+0.5;
  const X=v=>L+(Math.log2(v)-lx0)/(lx1-lx0)*pw,Y=v=>T+ph-(Math.log2(v)-ly0)/(ly1-ly0)*ph;
  for(let k=8;k<=ly1;k+=2){el("line",{x1:L,x2:L+pw,y1:Y(2**k),y2:Y(2**k),stroke:C.faint},svg);el("text",{x:L-6,y:Y(2**k)+4,"text-anchor":"end",fill:C.dim,"font-size":10,"font-family":"JetBrains Mono"},svg,"2"+sup(k));}
  const Ns=e.tasks[0].rows.map(r=>r.N);
  Ns.forEach(n=>{el("text",{x:X(n),y:T+ph+15,"text-anchor":"middle",fill:C.dim,"font-size":10,"font-family":"JetBrains Mono"},svg,(n/1000).toFixed(0)+"k");});
  el("path",{d:`M${X(n0)},${Y(n0)}L${X(n1)},${Y(n1)}`,stroke:C.ink,"stroke-dasharray":"5 4",fill:"none"},svg);
  el("text",{x:X(n1)-4,y:Y(n1)-6,"text-anchor":"end",fill:C.ink,"font-size":10},svg,"S = N (store everything)");
  el("path",{d:`M${X(n0)},${Y(2**8)}L${X(n1)},${Y(2**8)}`,stroke:C.dim,"stroke-dasharray":"2 3",fill:"none"},svg);
  el("text",{x:L+4,y:Y(2**8)-5,fill:C.dim,"font-size":10},svg,"flat = memory does not grow with N");
  e.tasks.forEach((t,i)=>{let d="";t.rows.forEach((r,k)=>{d+=(k?"L":"M")+X(r.N)+","+Y(r.best_S);});
    el("path",{d,fill:"none",stroke:cols[i],"stroke-width":2.4},svg);t.rows.forEach(r=>el("circle",{cx:X(r.N),cy:Y(r.best_S),r:3.5,fill:cols[i]},svg));
    const lx=[0,150,345][i]||i*160;el("rect",{x:L+lx-50,y:H-16,width:10,height:10,rx:2,fill:cols[i]},svg);el("text",{x:L+lx-36,y:H-7,fill:C.ink,"font-size":10},svg,t.task+(t.best_slope!=null?` (slope ${t.best_slope.toFixed(2)})`:""));});
  const q=e.tasks[0].rows;
  el("text",{x:L,y:12,fill:C.dim,"font-size":10},svg,"best one-pass classical sketch: S* vs N · quantum register: "+q.map(r=>r.qubits).join(" → ")+" index qubits");
  el("text",{x:L+pw/2,y:T+ph+30,"text-anchor":"middle",fill:C.dim,"font-size":10},svg,"N = common SNPs (nested random subsets of chr17–22)");
  /* table */
  const tb=document.getElementById("e8table");
  tb.innerHTML=`<tr><th>task</th><th>exact</th>`+Ns.map(n=>`<th>N=${(n/1000).toFixed(0)}k<br>S* (S*/N)</th>`).join("")+`<th>slope</th></tr>`;
  e.tasks.forEach(t=>{const big=t.rows[t.rows.length-1];tb.innerHTML+=`<tr><td>${t.task}</td><td class="n">${big.exact.toFixed(2)}</td>`+
    t.rows.map(r=>`<td class="n${r.best_S/r.N>=0.25?" gold":""}">${fmtS(r.best_S)}${r.stream_ok?"":"†"}<br>(${fmtR(r.best_S/r.N)})</td>`).join("")+`<td class="n">${t.best_slope==null?"—":t.best_slope.toFixed(2)}</td></tr>`;});
  /* title from the numbers: slope ~1 means classical memory grows like N */
  const sl=e.tasks.map(t=>t.best_slope).filter(v=>v!=null);const minSl=Math.min(...sl),maxSl=Math.max(...sl);
  const fracBig=Math.min(...e.tasks.map(t=>{const r=t.rows[t.rows.length-1];return r.best_S/r.N;}));
  let title;
  const fracMax=Math.max(...e.tasks.map(t=>{const r=t.rows[t.rows.length-1];return r.best_S/r.N;}));
  const sMin=Math.min(...e.tasks.map(t=>t.rows[t.rows.length-1].best_S)),sMax=Math.max(...e.tasks.map(t=>t.rows[t.rows.length-1].best_S));
  if(maxSl<0.75&&fracMax<=0.05) title=`Dense but compressible: <span class="gold">${fmtS(sMin)}–${fmtS(sMax)} classical counters</span> suffice at every N (≤ ${fmtR(fracMax)}·N, slope ≤ ${maxSl.toFixed(2)}) — no memory advantage here either`;
  else if(minSl>=0.75&&fracBig>=0.05) title=`Dense signal: classical memory grows <span class="gold">in proportion to N</span> (slope ${minSl.toFixed(2)}–${maxSl.toFixed(2)}), the quantum register only by log₂N — evidence, not proof`;
  else if(maxSl<=0.35) title=`Even with a dense signal, classical memory <span class="gold">stays flat as N grows</span> (slope ≤ ${maxSl.toFixed(2)}): the SNPs are redundant, so no advantage`;
  else title=`Dense signal, mixed scaling: classical S* grows with N at slope <span class="gold">${minSl.toFixed(2)}–${maxSl.toFixed(2)}</span> — between flat (0) and the paper's regime (1)`;
  document.getElementById("e8-title").innerHTML=title;
  setText("e8-note",`1000 Genomes phase 3, ${e.n_snps.toLocaleString()} common SNPs (chr17–22), 1,007 people (Europe + East Asia), amplitude features p = x²/‖x‖², no feature selection. S* = best of 6 one-pass sketches (count-sketch, hashing, sparse JL, Misra–Gries, reservoir, random SNP subset), stable within max(0.01, one test positive) of the exact centroid, 5 seeds; "slope" = log–log slope of S* vs N (1 = grows like N, 0 = flat). † = even the full-memory (S = N) stream at M = 100·N misses the band there, so that S* reflects too few records, not memory. Labels are sampled populations (not clusters of the data). Limits: ~31 test positives per task; one split; quantum = classical simulation. ${e.partial?"PARTIAL RESULTS. ":""}All numbers from results/e8_route2_genotypes.json.`);
})();

"""


def main():
    s = TPL.read_text()
    if MARK in s:
        print("[patch] template already contains the E8 slide; nothing to do")
        return
    anchor_html = '</div></div>\n<div id="progress"></div>'
    assert anchor_html in s
    s = s.replace(anchor_html, HTML + anchor_html)
    anchor_js = "/* ---- navigation ---- */"
    assert anchor_js in s
    s = s.replace(anchor_js, JS + anchor_js)
    TPL.write_text(s)
    print(f"[patch] inserted the E8 backup slide into {TPL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
