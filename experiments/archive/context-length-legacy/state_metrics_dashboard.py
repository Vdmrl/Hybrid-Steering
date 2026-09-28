"""Build the interactive state-metrics dashboard comparing steering-timing modes."""

import argparse
import base64
import json
from pathlib import Path

import torch
from plotly.offline import get_plotlyjs

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN steering-mode state metrics</title>
<script>__PLOTLY__</script>
<style>
:root{color-scheme:light;--bg:#f5f7fa;--panel:#fff;--panel2:#f8fafc;--text:#172033;--muted:#667085;--border:#dbe1ea;--accent:#3156c8;--shadow:0 8px 26px rgba(15,23,42,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{max-width:1500px;margin:auto;padding:24px}.header h1{margin:0 0 7px;font-size:26px;line-height:1.2}.header p{max-width:1000px;margin:0;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow)}.card{padding:14px 16px}.label{color:var(--muted);font-size:12px;font-weight:700}.value{margin-top:3px;font-size:23px;font-weight:760}.toolbar{display:flex;align-items:end;gap:12px;flex-wrap:wrap;padding:14px;margin-bottom:14px}.field{display:grid;gap:5px}.field label{font-size:12px;color:var(--muted);font-weight:700}select,button{min-height:38px;border:1px solid var(--border);border-radius:9px;background:var(--panel2);color:var(--text);padding:8px 11px;font:inherit}select:focus-visible,button:focus-visible{outline:3px solid rgba(49,86,200,.25);outline-offset:1px;border-color:var(--accent)}button{cursor:pointer;font-weight:700}.charts{display:grid;gap:14px}.chart-panel{padding:13px 14px 5px}.section-head{padding:2px 4px}.section-head h2{margin:0;font-size:17px}.section-head p{margin:3px 0 0;color:var(--muted);font-size:12px}.chart{height:420px}.method{margin-top:14px;padding:16px 18px;color:var(--muted)}.method strong{color:var(--text)}.error{padding:42px;text-align:center;color:#b42318}[hidden]{display:none!important}
@media(max-width:700px){.app{padding:13px}.cards{grid-template-columns:1fr 1fr}.toolbar{align-items:stretch}.field,select,button{width:100%}.chart{height:390px}.header h1{font-size:22px}}
</style>
</head>
<body>
<main class="app">
  <header class="header"><h1>GDN steering-mode state metrics</h1><p>Recurrent-state norm, rank, and cosine-to-previous, sampled at every response token under a steered, teacher-forced replay. Lines compare steering-timing modes (when delta is re-added) at a fixed scale and context length; position 0 is the first generated token. The prompt is a single un-sampled bulk forward call, matching how these responses were actually generated, so intra-prompt trajectories are not available -- only the boundary state right after the prompt.</p></header>
  <section class="cards" aria-label="Analysis summary">
    <article class="panel card"><div class="label">Steering modes</div><div class="value">__MODES__</div></article>
    <article class="panel card"><div class="label">Context lengths</div><div class="value">__LENGTHS__</div></article>
    <article class="panel card"><div class="label">GDN layers</div><div class="value">__LAYERS__</div></article>
    <article class="panel card"><div class="label">Examples per length</div><div class="value">__TEXTS__</div></article>
  </section>
  <section class="panel toolbar" aria-label="Metric filters">
    <div class="field"><label for="length">Context length</label><select id="length"></select></div>
    <div class="field"><label for="layer">GDN layer</label><select id="layer"><option value="-1">Average all layers</option></select></div>
    <div class="field"><label for="axis">Magnitude axes</label><select id="axis"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></div>
    <button id="reset" type="button">Reset view</button>
  </section>
  <div class="charts">
    <section class="panel chart-panel"><div class="section-head"><h2>Frobenius norm</h2><p>‖S<sub>t</sub>‖<sub>F</sub>, averaged over selected layers and heads.</p></div><div id="norm" class="chart" role="img" aria-label="State Frobenius norm by response token position"></div></section>
    <section class="panel chart-panel"><div class="section-head"><h2>Top singular value · σ<sub>1</sub></h2><p>Largest singular value of the state matrix.</p></div><div id="sigma1" class="chart" role="img" aria-label="Top singular value by response token position"></div></section>
    <section class="panel chart-panel"><div class="section-head"><h2>Effective rank</h2><p>exp(entropy of the normalized singular spectrum).</p></div><div id="rank" class="chart" role="img" aria-label="Effective rank by response token position"></div></section>
    <section class="panel chart-panel"><div class="section-head"><h2>Cosine to previous state</h2><p>cos(S<sub>t</sub>, S<sub>t-1</sub>); how much the state changes token to token.</p></div><div id="cosine" class="chart" role="img" aria-label="Cosine to previous state by response token position"></div></section>
  </div>
  <section class="panel method"><strong>Reading the boundary.</strong> Each mode presets S_0 = delta before the prompt except prompt_end, which starts natural and adds delta once, right after the prompt. repeated/periodic re-add delta every 1/N generated tokens respectively. Hover for the exact mean and contributing-example count.</section>
  <div id="error" class="panel error" hidden>The charts could not be rendered. Check the browser console for details.</div>
</main>
<script>
const PACKED=__DATA__;
const MODES=__MODES_JSON__;
const LENGTHS=__LENGTHS_JSON__;
const LAYERS=__LAYER_LABELS__;
const COLORS={"initial":"#2563eb","repeated":"#0891b2","periodic-32":"#059669","periodic-64":"#65a30d","periodic-128":"#d97706","prompt_end":"#dc2626"};
const METRIC_INDEX={"frobenius_norm":0,"sigma_1":1,"effective_rank":2,"stable_rank":3,"state_cosine":4};
const state={length:LENGTHS[0],layer:-1,axis:"log"};
const decoded={};
const byId=id=>document.getElementById(id);
function key(mode,length){return mode+"|"+length}
function values(mode,length){const k=key(mode,length);if(!(k in decoded)){const rec=PACKED[k];if(!rec){decoded[k]=null}else{const binary=atob(rec.values);const bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);decoded[k]={data:new Float32Array(bytes.buffer),shape:rec.shape,counts:rec.counts}}}return decoded[k]}
function series(mode,length,metric){const rec=values(mode,length);if(!rec)return null;const[positions,layers,metrics]=rec.shape,result=[];const l0=state.layer<0?0:state.layer,l1=state.layer<0?layers:state.layer+1;for(let p=0;p<positions;p++){let sum=0,n=0;for(let l=l0;l<l1;l++){const v=rec.data[((p*layers+l)*metrics)+metric];if(Number.isFinite(v)){sum+=v;n++}}result.push(n?sum/n:null)}return result}
function theme(){return {text:"#172033",muted:"#667085",border:"#dbe1ea"}}
function traces(metric){return MODES.filter(mode=>values(mode,state.length)).map(mode=>{const rec=values(mode,state.length),y=series(mode,state.length,metric);return {type:"scatter",mode:"lines",name:mode,x:rec.counts.map((_,i)=>i),y,line:{color:COLORS[mode]||"#475467",width:2},hovertemplate:`<b>${mode}</b><br>Position: %{x}<br>Mean: %{y:.5g}<extra></extra>`}})}
function layout(title,logAxis,showLegend){const c=theme();return {margin:{l:78,r:28,t:showLegend?52:20,b:58},paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:c.text,family:"Inter, ui-sans-serif, system-ui, sans-serif",size:13},legend:{orientation:"h",x:0,y:1.1,xanchor:"left",yanchor:"bottom"},xaxis:{title:"Response token position",gridcolor:c.border,zeroline:false},yaxis:{title,type:logAxis&&state.axis==="log"?"log":"linear",gridcolor:c.border,zeroline:false},hovermode:"closest",uirevision:`${state.length}-${state.layer}-${state.axis}`,showlegend:showLegend}}
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"],toImageButtonOptions:{filename:"gdn-state-metrics",format:"png",scale:2}};
function render(){Promise.all([Plotly.react("norm",traces(METRIC_INDEX.frobenius_norm),layout("Frobenius norm",true,true),config),Plotly.react("sigma1",traces(METRIC_INDEX.sigma_1),layout("Sigma 1",true,false),config),Plotly.react("rank",traces(METRIC_INDEX.effective_rank),layout("Effective rank",false,false),config),Plotly.react("cosine",traces(METRIC_INDEX.state_cosine),layout("Cosine",false,false),config)]).catch(error=>{console.error(error);byId("error").hidden=false})}
function init(){LENGTHS.forEach(length=>byId("length").add(new Option(length===0?"tweet-only":`${length} tokens`,length)));LAYERS.forEach((label,index)=>byId("layer").add(new Option(label,index)));byId("length").addEventListener("change",event=>{state.length=Number(event.target.value);render()});byId("layer").addEventListener("change",event=>{state.layer=Number(event.target.value);render()});byId("axis").addEventListener("change",event=>{state.axis=event.target.value;render()});byId("reset").addEventListener("click",()=>{Object.assign(state,{length:LENGTHS[0],layer:-1,axis:"log"});byId("length").value=String(LENGTHS[0]);byId("layer").value="-1";byId("axis").value="log";render()});render()}
window.addEventListener("error",event=>{console.error("Dashboard error",event.error||event.message);byId("error").hidden=false});init();
</script>
</body>
</html>
"""

METRIC_ORDER = ("frobenius_norm", "sigma_1", "effective_rank", "stable_rank", "state_cosine")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dirs) != len(args.labels):
        raise ValueError("--run-dirs and --labels must have the same length")

    packed = {}
    layer_labels = None
    lengths: set[int] = set()
    max_examples = 0
    for run_dir, label in zip(args.run_dirs, args.labels, strict=True):
        state_dir = run_dir / "intermediate" / "state_metrics"
        for path in sorted(state_dir.glob("length-*.pt")):
            artifact = torch.load(path, weights_only=True)
            length = artifact["length"]
            lengths.add(length)
            max_examples = max(max_examples, artifact["examples"])
            if layer_labels is None:
                layer_labels = [
                    f"GDN {index} · decoder {decoder}"
                    for index, decoder in enumerate(artifact["decoder_layers"])
                ]
            names = list(artifact["metric_names"])
            stacked = torch.stack(
                [artifact["response_mean"][name] for name in METRIC_ORDER], dim=-1
            )
            counts = artifact["response_count"][names[0]]
            values = stacked.contiguous().numpy()
            packed[f"{label}|{length}"] = {
                "shape": list(values.shape),
                "counts": counts.tolist(),
                "values": base64.b64encode(values.tobytes()).decode(),
            }

    html = (
        HTML.replace("__PLOTLY__", get_plotlyjs())
        .replace("__DATA__", json.dumps(packed, separators=(",", ":")))
        .replace("__MODES_JSON__", json.dumps(args.labels))
        .replace("__LENGTHS_JSON__", json.dumps(sorted(lengths)))
        .replace("__LAYER_LABELS__", json.dumps(layer_labels or []))
        .replace("__MODES__", str(len(args.labels)))
        .replace("__LENGTHS__", str(len(lengths)))
        .replace("__LAYERS__", str(len(layer_labels or [])))
        .replace("__TEXTS__", str(max_examples))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
