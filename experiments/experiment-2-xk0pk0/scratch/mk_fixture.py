"""Synthetic analysis tables with the real schema, to exercise src/figures.py."""
import json, math, random
from pathlib import Path
random.seed(7)
out = Path("scratch/fixture_results"); out.mkdir(parents=True, exist_ok=True)

def ci(p, w=0.004, n=295):
    return {"point": p, "lo": p - w, "hi": p + w, "n": n}

CONDS = [("adamw","native",2e-4,1.3345,True),("sgd","native",0.01,1.2146,True),
         ("sgd","scale_matched",0.0583,1.1802,False),("muon","native",0.02,1.3021,True),
         ("muon","scale_matched",0.00243,1.3288,True)]
rows=[]
for c,l,lr,ov,tr in CONDS:
    r={"run":f"{c}_{l}","condition":c,"label":l,"lr":lr,"n_steps":300,
       "n_steps_used":295,"warmup_dropped":5,"config":{},
       "loss_first":3.4,"loss_last5_mean":1.4 if tr else 3.41,
       "grad_norm_preclip_median":3.3,"grad_norm_preclip_max":9.0,
       "clip_fired_fraction":1.0 if tr else 0.6,
       "clip_fired_every_step":tr,"clip_fired_first_100":100 if tr else 60,
       "trajectory":{"loss_first20_mean":3.4,"loss_last20_mean":1.4 if tr else 3.4,
                     "loss_drop":2.0 if tr else 0.001,"loss_min":1.2,"loss_max":3.5,
                     "all_losses_finite":True,"grad_norm_preclip_median":3.3,
                     "threshold_loss_drop":0.10,
                     "verdict":"trains" if tr else "does_not_train"}}
    for s in ("whole","bf16_recorded","bf16_true","muon_covered"):
        r[f"{s}_overshoot"]=ci(ov+random.uniform(-0.002,0.002))
        r[f"{s}_zero_fraction"]=ci(0.93); r[f"{s}_intent_norm"]=ci(0.657)
        r[f"{s}_delivery"]=ci(0.45)
    for s in ("bf16_true","muon_covered"):
        for f in ("x_median","zero_step_fraction","abs_step_ge_2_share_of_moved",
                  "mean_abs_step_of_moved","ratio_predicted_from_x","decay_x_max"):
            r[f"geom_{s}_{f}"]=ci(0.1)
    rows.append(r)
(out/"by_condition.json").write_text(json.dumps(
    {"recorded_reference":{"uniform_x_reference":4/3},"warmup_dropped":5,"rows":rows}))

MULT=[0.03125,0.0625,0.125,0.25,0.5,1,2,4,8]
BASE={"adamw":1.3345,"sgd":1.2900,"muon":1.3180}
def mk_src(key, cond, lr, jitter):
    curve={}
    for o,b in BASE.items():
        pts=[]
        for m in MULT:
            # overshoot rises with magnitude and saturates near 4/3
            v=1.0+(b-1.0)*(m/(m+0.35))+jitter
            pts.append({"multiplier":m,"intent_norm_median":0.657*m,
                        "overshoot":ci(v,0.003),"zero_fraction":ci(0.9)})
        curve[o]=pts
    at={o:next(p for p in pts if p["multiplier"]==1.0)["overshoot"] for o,pts in curve.items()}
    paired={}
    for o in ("muon","sgd"):
        d=at[o]["point"]-at["adamw"]["point"]
        paired[f"{o}_minus_adamw"]={**ci(d,0.002),"n_steps_negative":295 if d<0 else 0,"n_steps":295}
    return {"source_run":key,"host_optimizer":cond,"host_lr":lr,"n_steps":295,
            "warmup_dropped":5,"multipliers":MULT,"curve":curve,
            "at_adamw_magnitude":at,"paired_difference_at_adamw_magnitude":paired,
            "self_check":{"n_steps":295,"max_abs_gap":0.0028,"all_ok":True}}
sources={f"{c}_{l}":mk_src(f"{c}_{l}",c,lr,random.uniform(-0.004,0.004))
         for c,l,lr,_,_ in CONDS}
across={}
for o in ("sgd","muon"):
    k=f"{o}_minus_adamw"
    per={s:c["paired_difference_at_adamw_magnitude"][k] for s,c in sources.items()}
    across[k]={"per_trajectory":per,"n_trajectories":len(per),
               "n_trajectories_negative":sum(v["point"]<0 for v in per.values()),
               "n_trajectories_interval_excludes_zero":sum(v["hi"]<0 or v["lo"]>0 for v in per.values()),
               "min_point":min(v["point"] for v in per.values()),
               "max_point":max(v["point"] for v in per.values())}
(out/"shadow_curve.json").write_text(json.dumps(
    {"primary":"adamw_native","sources":sources,"across_trajectories":across,
     **sources["adamw_native"]}))

EDGES=[round(-9.0+0.02*i,6) for i in range(651)]
h={}
for c,l,lr,ov,tr in CONDS:
    counts=[int(2e6*math.exp(-((EDGES[i]+1.2)**2)/0.9))+1 for i in range(650)]
    sh=[0.60,0.24,0.09,0.04,0.015,0.008,0.004,0.002,0.0006,0.0002,0.0001,0.00005]
    tot=1_000_000
    h[f"{c}_{l}"]={"step":300,"edges":list(range(13)),
        "counts":[int(s*tot) for s in sh],"above_edges":100,"total":tot,
        "share":sh,"zero_step_fraction":0.60,"abs_step_ge_2_share_of_moved":0.36,
        "mean_abs_step_of_moved":1.7,
        "x_log10_hist":{"edges":EDGES,"counts":counts,"below":10,"above":10},
        "x_quantiles_from_hist":{"0.5":0.0757},"chi_square_vs_uniform":3.28e6,
        "ratio_predicted_from_x":ov,"ratio_from_realized_steps":ov}
(out/"step_histograms.json").write_text(json.dumps(h))
print("fixture written to", out)
