"""Insert the Route 2 (E7) backup slide into presentation/plan3_template.html (idempotent).
The slide draws everything from RESULTS.e7, which scripts/08_slides.py reads from results/e7_route2.json.
Run:  python scripts/09b_patch_template_e7.py && python scripts/08_slides.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import ROOT

TPL = ROOT / "presentation" / "plan3_template.html"
MARK = "<!-- B4 · ROUTE 2: CLASSICAL MEMORY (E7) -->"

HTML = MARK + """
<section class="slide">
  <div class="step backup"><span class="num">B</span>Backup · Route 2: classical memory stress test (E7)</div>
  <h2 id="e7-title" style="font-size:29px"></h2>
  <div class="row">
    <div class="col" style="flex:1.15"><svg id="e7plot" viewBox="0 0 640 330" width="100%"></svg></div>
    <div class="col" style="gap:8px;justify-content:flex-start">
      <style>#e7table th,#e7table td{padding:6px 8px}</style>
      <table id="e7table" style="font-size:13px"></table>
      <p class="note" id="e7-note"></p>
    </div>
  </div>
</section>

"""

JS = """/* ---- B4: E7 route 2, classical memory ---- */
(function(){
  const e=R.e7; if(!e) return;
  const blk=e.by_K[e.by_K.length-1];
  const sup=n=>String(n).replace(/-/g,"⁻").replace(/\\d/g,d=>"⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  const fmtS=v=>v==null?"never":(v>=64&&Number.isInteger(Math.log2(v))?"2"+sup(Math.log2(v)):v.toLocaleString());
  const fmtR=v=>{if(v==null)return"—";if(v>=0.1)return v.toFixed(2);const e=Math.floor(Math.log10(v));const m=Math.round(v/10**e);return (m>=10?"1·10"+sup(e+1):m+"·10"+sup(e));};
  const tasks=blk.real.concat(blk.synthetic?[blk.synthetic]:[]);
  const show=["count_sketch","hashing","jl_sparse","heavy_hitters_mg","reservoir"];
  const cols={count_sketch:C.ink,hashing:C.g3,jl_sparse:C.g2,heavy_hitters_mg:C.gold,reservoir:C.red};
  const names={count_sketch:"count-sketch",hashing:"feature hashing",jl_sparse:"sparse JL",heavy_hitters_mg:"heavy hitters (MG)",reservoir:"reservoir"};
  const onePass=k=>!["hashing_logreg","heavy_hitters_top","jl_dense"].includes(k);
  const bestRatio=t=>Math.min(...Object.entries(t.methods).filter(([k])=>onePass(k)).map(([,m])=>m.S_star_cell_stable==null?1:m.S_star_cell_stable/t.N));
  /* chart: B cells (the class with a real test set) and the no-structure control */
  const svg=document.getElementById("e7plot");const W=640,H=330,T=34,B=54;
  const bcell=blk.real.find(t=>t.positive==="B cells")||blk.real[0];
  const panels=[[bcell.positive+" (real)",bcell]].concat(blk.synthetic?[["synthetic control: no structure",blk.synthetic]]:[]);
  panels.forEach(([title,t],pi)=>{
    const L=52+pi*320,pw=270,ph=H-T-B,N=t.N;
    const lx0=3,lx1=Math.log2(N);const X=v=>L+(Math.log2(v)-lx0)/(lx1-lx0)*pw;const Y=v=>T+ph-(v-0.5)/0.5*ph;
    [0.5,0.6,0.7,0.8,0.9,1].forEach(v=>{el("line",{x1:L,x2:L+pw,y1:Y(v),y2:Y(v),stroke:C.faint},svg);if(pi===0)el("text",{x:L-6,y:Y(v)+4,"text-anchor":"end",fill:C.dim,"font-size":11,"font-family":"JetBrains Mono"},svg,v.toFixed(1));});
    for(let k=4;k<=lx1;k+=4) el("text",{x:X(2**k),y:H-B+14,"text-anchor":"middle",fill:C.dim,"font-size":10,"font-family":"JetBrains Mono"},svg,"2"+sup(k));
    const ex=t.exact.balanced_accuracy,tol=t.tolerance_cell||0.01;
    el("rect",{x:L,y:Y(Math.min(1,ex+tol)),width:pw,height:Y(ex-tol)-Y(Math.min(1,ex+tol)),fill:C.ink,opacity:.08},svg);
    el("line",{x1:L,x2:L+pw,y1:Y(ex),y2:Y(ex),stroke:C.ink,"stroke-dasharray":"2 3"},svg);
    el("text",{x:L+pw-2,y:ex>0.93?Y(ex)+13:Y(ex)-4,"text-anchor":"end",fill:C.ink,"font-size":10},svg,"exact centroid "+ex.toFixed(2)+" ± "+tol.toFixed(2));
    const q=blk.quantum.index_qubits;
    el("line",{x1:X(q),x2:X(q),y1:T,y2:T+ph,stroke:C.red,"stroke-dasharray":"4 3","stroke-width":1.2},svg);
    el("text",{x:X(q)+4,y:T+ph-6,fill:C.red,"font-size":10},svg,"quantum: "+q+" index qubits");
    el("text",{x:X(N)-2,y:T+ph-6,"text-anchor":"end",fill:C.dim,"font-size":10},svg,"S = N");
    show.forEach(k=>{const m=t.methods[k];if(!m)return;let d="";m.rows.forEach((r,i)=>{d+=(i?"L":"M")+X(r.S)+","+Y(Math.max(0.5,r.ba));});
      el("path",{d,fill:"none",stroke:cols[k],"stroke-width":2.2,opacity:.95},svg);m.rows.forEach(r=>el("circle",{cx:X(r.S),cy:Y(Math.max(0.5,r.ba)),r:2.4,fill:cols[k]},svg));});
    el("text",{x:L,y:14,fill:C.ink,"font-size":12,"font-weight":700},svg,title);
    el("text",{x:L,y:27,fill:C.dim,"font-size":10},svg,"N = "+N.toLocaleString()+" · M = "+t.M_run_over_N+"·N records · π̂ = "+t.pi_hat.toFixed(3)+" · test positives "+t.n_test_pos);
  });
  el("text",{x:52,y:H-24,fill:C.dim,"font-size":10},svg,"x: classical memory S (counters, log scale) · y: balanced accuracy on the test split, 5 seeds · band: exact centroid ± tol");
  show.forEach((k,i)=>{const x=52+[0,105,225,310,445][i],y=H-12;el("rect",{x,y,width:10,height:10,rx:2,fill:cols[k]},svg);el("text",{x:x+14,y:y+9,fill:C.ink,"font-size":10},svg,names[k]);});
  /* table */
  const tb=document.getElementById("e7table");
  const tshow=["count_sketch","hashing","heavy_hitters_mg","reservoir"];
  tb.innerHTML=`<tr><th>K=${blk.K} · N=${sci(blk.N)}</th><th>exact<br>(tol)</th>`+tshow.map(k=>`<th>${names[k]}<br>S* (S*/N)</th>`).join("")+`</tr>`;
  tasks.forEach(t=>{tb.innerHTML+=`<tr><td>${t.positive}</td><td class="n">${t.exact.balanced_accuracy.toFixed(2)}<br>(${(t.tolerance_cell||0.01).toFixed(2)})</td>`+tshow.map(k=>{const m=t.methods[k];if(!m)return"<td></td>";const s=m.S_star_cell_stable;return `<td class="n${s!=null&&s/t.N<=0.01?" gold":""}">${fmtS(s)}<br>(${fmtR(s==null?null:s/t.N)})</td>`;}).join("")+`</tr>`;});
  /* title + note */
  const worst=Math.max(...blk.real.map(bestRatio));
  const synBest=blk.synthetic?bestRatio(blk.synthetic):null;
  const synTxt=synBest==null||synBest>=0.5?"≈ N":"≈ "+fmtR(synBest)+"·N";
  let title;
  if(worst<=0.01) title=`No memory advantage on PBMC3k: <span class="gold">S ≈ ${fmtR(worst)}·N</span> classical counters match the exact centroid — the gap exists only without structure (control: S* ${synTxt})`;
  else if(worst>=0.25) title=`Evidence for the claim on real data: every classical sketch needs <span class="gold">S ≳ ${fmtR(worst)}·N</span> to match the exact centroid on PBMC3k gene pairs`;
  else title=`No exponential gap on PBMC3k: classical sketches need <span class="gold">S ≈ ${fmtR(worst)}·N</span>; the control with no structure needs S* ${synTxt}`;
  document.getElementById("e7-title").innerHTML=title;
  const lr=blk.real.map(t=>t.logreg_full?`${t.positive.split(" ")[0]} ${t.logreg_full.balanced_accuracy.toFixed(2)}`:null).filter(Boolean).join(", ");
  setText("e7-note",`S* = smallest S that stays within tol = max(0.01, one test positive) of the exact centroid for all larger S (5 seeds; strict 0.01 and AUPRC variants in the JSON). Same records, ±1/π̂ weights and split for all methods. Quantum sketch at this N: ${blk.quantum.index_qubits} index qubits (≈${blk.quantum.logical_qubits} logical), classical simulation, no hardware. Logreg on all N features: ${lr}. Limits: Louvain labels are circular; DC/Mk test sets = 11/5 cells; one split. ${e.partial?"PARTIAL RESULTS. ":""}All numbers from results/e7_route2.json.`);
})();

"""


def main():
    s = TPL.read_text()
    if MARK in s:
        print("[patch] template already contains the E7 slide; nothing to do")
        return
    anchor_html = '</div></div>\n<div id="progress"></div>'
    assert anchor_html in s, "html anchor not found"
    s = s.replace(anchor_html, HTML + anchor_html)
    anchor_js = "/* ---- navigation ---- */"
    assert anchor_js in s, "js anchor not found"
    s = s.replace(anchor_js, JS + anchor_js)
    TPL.write_text(s)
    print(f"[patch] inserted the E7 backup slide into {TPL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
