"""Build the interactive rank-one propagation dashboard."""

import argparse
import base64
import json
from pathlib import Path

import torch
from plotly.offline import get_plotlyjs

from gdn_interp.dynamics import measurement_positions

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN steering propagation</title>
<script>__PLOTLY__</script>
<style>
:root{color-scheme:light;--bg:#f5f7fa;--panel:#fff;--panel2:#f8fafc;--text:#172033;--muted:#667085;--border:#dbe1ea;--accent:#3156c8;--shadow:0 8px 26px rgba(15,23,42,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{max-width:1500px;margin:auto;padding:24px}.header h1{margin:0 0 7px;font-size:26px;line-height:1.2}.header p{max-width:1000px;margin:0;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow)}.card{padding:14px 16px}.label{color:var(--muted);font-size:12px;font-weight:700}.value{margin-top:3px;font-size:23px;font-weight:760}.toolbar{display:flex;align-items:end;gap:12px;flex-wrap:wrap;padding:14px;margin-bottom:14px}.field{display:grid;gap:5px}.field label{font-size:12px;color:var(--muted);font-weight:700}select,button{min-height:38px;border:1px solid var(--border);border-radius:9px;background:var(--panel2);color:var(--text);padding:8px 11px;font:inherit}select:focus-visible,button:focus-visible{outline:3px solid rgba(49,86,200,.25);outline-offset:1px;border-color:var(--accent)}button{cursor:pointer;font-weight:700}.charts{display:grid;gap:14px}.chart-panel{padding:13px 14px 5px}.section-head{padding:2px 4px}.section-head h2{margin:0;font-size:17px}.section-head p{margin:3px 0 0;color:var(--muted);font-size:12px}.chart{height:420px}.method{margin-top:14px;padding:16px 18px;color:var(--muted)}.method strong{color:var(--text)}.error{padding:42px;text-align:center;color:#b42318}[hidden]{display:none!important}
@media(max-width:700px){.app{padding:13px}.cards{grid-template-columns:1fr 1fr}.toolbar{align-items:stretch}.field,select,button{width:100%}.chart{height:390px}.header h1{font-size:22px}}
</style>
</head>
<body>
<main class="app">
  <header class="header"><h1>GDN steering propagation</h1><p>Rank-one scale-1.25 steering followed through the steered, teacher-forced prompt and saved response. Token zero is the first generated token; negative positions belong to the prompt.</p></header>
  <section class="cards" aria-label="Analysis summary">
    <article class="panel card"><div class="label">Saved generations</div><div class="value">__TEXTS__</div></article>
    <article class="panel card"><div class="label">Context lengths</div><div class="value">48–2,048</div></article>
    <article class="panel card"><div class="label">GDN layers</div><div class="value">__LAYERS__</div></article>
    <article class="panel card"><div class="label">Heads per layer</div><div class="value">__HEADS__</div></article>
  </section>
  <section class="panel toolbar" aria-label="Metric filters">
    <div class="field"><label for="layer">GDN layer</label><select id="layer"><option value="-1">Average all layers</option></select></div>
    <div class="field"><label for="head">Value head</label><select id="head"><option value="-1">Average all heads</option></select></div>
    <div class="field"><label for="axis">Magnitude axes</label><select id="axis"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></div>
    <button id="reset" type="button">Reset view</button>
  </section>
  <div class="charts">
    <section class="panel chart-panel"><div class="section-head"><h2>Effective singular magnitude · σ<sub>t</sub></h2><p>σ<sub>0</sub>‖u<sub>t</sub>‖ for the propagated leading rank-one component.</p></div><div id="sigma" class="chart" role="img" aria-label="Effective singular magnitude by token position"></div></section>
    <section class="panel chart-panel"><div class="section-head"><h2>Query-detector cosine</h2><p>cos(u<sub>t</sub>, q<sub>t</sub>); sign controls the direction of the readout.</p></div><div id="cosine" class="chart" role="img" aria-label="Cosine between propagated left singular vector and query"></div></section>
    <section class="panel chart-panel"><div class="section-head"><h2>Norm product</h2><p>‖u<sub>t</sub>‖‖q<sub>t</sub>‖ before multiplication by σ<sub>0</sub> and the cosine.</p></div><div id="norm" class="chart" role="img" aria-label="Product of propagated vector and query norms"></div></section>
  </div>
  <section class="panel method"><strong>Reading the boundary.</strong> The dashed vertical line marks the first response token. Prompt positions are sampled for display; every generated-token position remains available. Hover for the exact mean and contributing-text count. The raw artifact retains every token, layer, head, and text.</section>
  <div id="error" class="panel error" hidden>The charts could not be rendered. Check the browser console for details.</div>
</main>
<script>
const PACKED=__DATA__;
const LAYERS=__LAYER_LABELS__;
const HEADS=__HEADS_JSON__;
const COLORS=["#2563eb","#0891b2","#059669","#65a30d","#d97706","#dc2626","#7c3aed"];
const state={layer:-1,head:-1,axis:"log"};
const decoded={};
const byId=id=>document.getElementById(id);
function values(length){if(!decoded[length]){const binary=atob(PACKED[length].values);const bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);decoded[length]=new Float32Array(bytes.buffer)}return decoded[length]}
function series(record,metric){const data=values(record.length),[positions,layers,heads,metrics]=record.shape,result=[];for(let p=0;p<positions;p++){let sum=0,n=0;const l0=state.layer<0?0:state.layer,l1=state.layer<0?layers:state.layer+1,h0=state.head<0?0:state.head,h1=state.head<0?heads:state.head+1;for(let l=l0;l<l1;l++)for(let h=h0;h<h1;h++){const value=data[(((p*layers+l)*heads+h)*metrics)+metric];if(Number.isFinite(value)){sum+=value;n++}}result.push(n?sum/n:null)}return result}
function theme(){return {text:"#172033",muted:"#667085",border:"#dbe1ea"}}
function traces(metric){return Object.values(PACKED).sort((a,b)=>a.length-b.length).map((record,index)=>({type:"scatter",mode:"lines",name:`${record.length} tokens`,x:record.positions,y:series(record,metric),customdata:record.counts.map((n,i)=>[record.length,n,record.positions[i]<0?"Prompt":"Generated"]),line:{color:COLORS[index],width:2},hovertemplate:"<b>%{customdata[0]}-token context</b><br>%{customdata[2]} position: %{x}<br>Mean: %{y:.5g}<br>Texts: %{customdata[1]}<extra></extra>"}))}
function layout(title,logAxis,showLegend){const c=theme();return {margin:{l:78,r:28,t:showLegend?68:28,b:58},paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:c.text,family:"Inter, ui-sans-serif, system-ui, sans-serif",size:13},legend:{orientation:"h",x:0,y:1.13,xanchor:"left",yanchor:"bottom",title:{text:"Context length · "}},xaxis:{title:"Token position relative to generation boundary",gridcolor:c.border,zeroline:false},yaxis:{title,type:logAxis&&state.axis==="log"?"log":"linear",gridcolor:c.border,zeroline:false},hovermode:"closest",uirevision:`${state.layer}-${state.head}-${state.axis}`,shapes:[{type:"line",x0:0,x1:0,y0:0,y1:1,yref:"paper",line:{color:"#475467",width:2,dash:"dash"}}],annotations:[{x:0,y:1,xref:"x",yref:"paper",text:"Generation begins",showarrow:false,xanchor:"left",yanchor:"bottom",font:{size:12,color:c.muted},bgcolor:"rgba(255,255,255,.85)"}],showlegend:showLegend}}
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"],toImageButtonOptions:{filename:"gdn-propagation",format:"png",scale:2}};
function render(){Promise.all([Plotly.react("sigma",traces(0),layout("Effective singular magnitude",true,true),config),Plotly.react("cosine",traces(1),layout("Cosine",false,false),config),Plotly.react("norm",traces(2),layout("Norm product",true,false),config)]).catch(error=>{console.error(error);byId("error").hidden=false})}
function init(){LAYERS.forEach((label,index)=>byId("layer").add(new Option(label,index)));HEADS.forEach(head=>byId("head").add(new Option(`Head ${head}`,head)));byId("layer").addEventListener("change",event=>{state.layer=Number(event.target.value);render()});byId("head").addEventListener("change",event=>{state.head=Number(event.target.value);render()});byId("axis").addEventListener("change",event=>{state.axis=event.target.value;render()});byId("reset").addEventListener("click",()=>{Object.assign(state,{layer:-1,head:-1,axis:"log"});byId("layer").value="-1";byId("head").value="-1";byId("axis").value="log";render()});render()}
window.addEventListener("error",event=>{console.error("Dashboard error",event.error||event.message);byId("error").hidden=false});init();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    propagation = args.run_dir / "intermediate" / "propagation"
    artifact = torch.load(propagation / "averages.pt", weights_only=True)
    packed = {}
    texts = 0
    for length, group in artifact["by_length"].items():
        prompt = group["prompt_tokens"]
        total = len(group["count"])
        indices = [position - 1 for position in measurement_positions(prompt)]
        indices += list(range(prompt, total))
        values = group["mean"][indices].contiguous().numpy()
        counts = group["count"][indices].tolist()
        texts = max(texts, max(counts))
        packed[str(length)] = {
            "length": length,
            "shape": list(values.shape),
            "positions": [index - prompt for index in indices],
            "counts": counts,
            "values": base64.b64encode(values.tobytes()).decode(),
        }
    layer_labels = [
        f"GDN {index} · decoder {decoder}"
        for index, decoder in enumerate(artifact["decoder_layers"])
    ]
    html = (
        HTML.replace("__PLOTLY__", get_plotlyjs())
        .replace("__DATA__", json.dumps(packed, separators=(",", ":")))
        .replace("__LAYER_LABELS__", json.dumps(layer_labels))
        .replace("__HEADS_JSON__", json.dumps(artifact["heads"]))
        .replace("__TEXTS__", f"{texts:,}")
        .replace("__LAYERS__", str(len(layer_labels)))
        .replace("__HEADS__", str(len(artifact["heads"])))
    )
    output = args.run_dir / "plots" / "propagation.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
