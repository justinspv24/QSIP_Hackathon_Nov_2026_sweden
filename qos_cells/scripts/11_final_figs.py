import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "/inter/" in f.lower(): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Inter","mathtext.fontset":"cm","font.size":22,
  "axes.edgecolor":"#C9C7C1","axes.labelcolor":"#4A4A48","xtick.color":"#4A4A48","ytick.color":"#4A4A48",
  "axes.spines.top":False,"axes.spines.right":False})
BG="#FFFFFF"; INK="#08090B"; MUTE="#C9C7C1"; BLUE="#08090B"; ORANGE="#FFC900"; DIM="#4A4A48"
OUT="/home/claude/final/fig/"
def save(fig,name):
    fig.savefig(OUT+name,dpi=200,facecolor=BG,bbox_inches="tight",pad_inches=0.15); plt.close(fig)

# A: class counts
cls=[("CD4 T cells",1144),("CD14 monocytes",480),("B cells",342),("CD8 T cells",316),("NK cells",154),("FCGR3A monocytes",150),("Dendritic cells",37),("Megakaryocytes",15)]
fig,ax=plt.subplots(figsize=(8.6,5.6),facecolor=BG); ax.set_facecolor(BG)
names=[c[0] for c in cls][::-1]; vals=[c[1] for c in cls][::-1]
cols=[ORANGE if v<100 else MUTE for v in vals]
ax.barh(names,vals,color=cols,height=0.68); ax.set_xscale("log"); ax.set_xlim(8,4000)
for i,v in enumerate(vals):
    lab=f"{v:,}"+(f"  ({v/2638*100:.1f} %)" if v<100 else "")
    ax.text(v*1.12,i,lab,va="center",fontsize=20,color=INK if v<100 else DIM,fontweight="bold" if v<100 else "normal")
ax.set_xlabel("cells per type (log scale)",fontsize=20); ax.tick_params(axis="y",length=0,labelsize=20)
ax.spines["left"].set_visible(False)
save(fig,"classes.png")

# B: rare-class sample cost
pis=["0.5","0.1","0.02","0.005"]; meas=[1.0,3.0608,13.484,71.185]; law=[1.0,2.7778,12.755,50.251]
import numpy as np
x=np.arange(4); w=0.36
fig,ax=plt.subplots(figsize=(8.6,5.8),facecolor=BG); ax.set_facecolor(BG)
ax.bar(x-w/2,law,w,color=MUTE,label=r"prediction  $1/[\pi(1-\pi)]$")
ax.bar(x+w/2,meas,w,color=ORANGE,edgecolor=INK,linewidth=1.2,label="measured (simulation)")
ax.set_yscale("log"); ax.set_ylim(0.7,400)
for i in range(4):
    ax.text(x[i]-w/2,law[i]*1.15,f"×{law[i]:.0f}" if law[i]>=10 else f"×{law[i]:.1f}",ha="center",fontsize=18,color=DIM)
    ax.text(x[i]+w/2,meas[i]*1.15,f"×{meas[i]:.0f}" if meas[i]>=10 else f"×{meas[i]:.1f}",ha="center",fontsize=18,color=INK,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([rf"$\pi = {p}$" for p in pis],fontsize=24)
ax.set_ylabel("extra samples needed\n(vs. a 50 / 50 class)",fontsize=19)
ax.set_yticks([1,10,100]); ax.set_yticklabels(["×1","×10","×100"])
ax.legend(frameon=False,fontsize=19,loc="upper left")
save(fig,"rarecost.png")

# C: hardware, N=4, 8 test cells
cells=[ # label, formula, shots, garnet raw
 (1,0.2001,0.219,0.163),(1,0.1043,0.109,0.140),(1,0.2130,0.247,0.152),(1,0.2166,0.241,0.171),
 (0,-0.9036,-0.905,-0.684),(0,-0.8135,-0.83,-0.566),(0,-0.9029,-0.9,-0.624),(0,-0.4879,-0.495,-0.399)]
thr=-0.14405
fig,ax=plt.subplots(figsize=(10.4,5.9),facecolor=BG); ax.set_facecolor(BG)
x=np.arange(8); w=0.26
ax.bar(x-w,[c[1] for c in cells],w,color=MUTE,label="classical (exact formula)")
ax.bar(x,[c[2] for c in cells],w,color="#4A4A48",label="quantum simulator")
ax.bar(x+w,[c[3] for c in cells],w,color=ORANGE,edgecolor=INK,linewidth=1.0,label="real IQM Garnet (raw)")
ax.axhline(0,color="#C9C7C1",lw=1)
ax.axhline(thr,color=INK,lw=1.6,ls=(0,(5,4)))
ax.text(-0.45,thr+0.03,"decision line",ha="left",fontsize=17,color=INK)
ax.set_xticks(x); ax.set_xticklabels([str(i+1) for i in range(8)],fontsize=18)
ax.text(1.5,-1.32,"4 rare cells (B cells)",ha="center",fontsize=20,color=INK,fontweight="bold",transform=ax.transData,clip_on=False)
ax.text(5.5,-1.32,"4 common cells",ha="center",fontsize=20,color=DIM,transform=ax.transData,clip_on=False)
ax.axvline(3.5,color="#C9C7C1",lw=1,ls=":")
ax.set_ylabel(r"classifier score  $\langle Z\rangle$",fontsize=20); ax.set_ylim(-1.05,0.55)
ax.legend(frameon=False,fontsize=18,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.13))
save(fig,"hardware.png")

# D: memory needed
rows=[ # label, N, sketch S*
 ("Blood cells\ngene pairs",2098176,256),
 ("Human genomes\nSNPs",791251,16384),
 ("Synthetic, no structure\n(worst case)",2098176,262144)]
fig,ax=plt.subplots(figsize=(10.4,5.6),facecolor=BG); ax.set_facecolor(BG)
y=np.arange(3)[::-1]; h=0.34
ax.barh(y+h/2,[r[1] for r in rows],h,color=MUTE,label="store every feature")
ax.barh(y-h/2,[r[2] for r in rows],h,color=ORANGE,edgecolor=INK,linewidth=1.0,label="best classical streaming sketch")
ax.set_xscale("log"); ax.set_xlim(50,3e8)
def fmt(v): return f"{v/1e6:.1f} M" if v>=1e6 else (f"{v/1e3:.0f} k" if v>=1e4 else f"{v:,}")
for yi,r in zip(y,rows):
    ax.text(r[1]*1.25,yi+h/2,fmt(r[1]),va="center",fontsize=19,color=DIM)
    ax.text(r[2]*1.25,yi-h/2,fmt(r[2])+f"   ({r[2]/r[1]*100:.2g} %)",va="center",fontsize=19,color=INK,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=19); ax.tick_params(axis="y",length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("numbers held in memory (log scale)",fontsize=19)
ax.legend(frameon=False,fontsize=18,loc="upper center",ncol=2,bbox_to_anchor=(0.45,1.14))
save(fig,"memory.png")
print("ok")
