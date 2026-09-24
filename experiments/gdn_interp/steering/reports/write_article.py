"""Build the experimental narrative from saved JSONL results (no inference)."""
import json
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/steering'


def read(folder, name):
    return [json.loads(line) for line in (DATA / folder / (name + '.jsonl')).read_text().splitlines()]


def rate(rows):
    return sum(r['detected_language'] == 'ru' for r in rows) / len(rows)


def main():
    full = read('confirmation/full', 'r0_all_p0_s2')
    rank1 = read('confirmation/rank1', 'r1_all_p0_s4')
    baseline = read('confirmation/full', 'r0_all_p0_s0')
    assert len(full) == len(rank1) == len(baseline) == 20
    assert [r['question'] for r in full] == [r['question'] for r in baseline]
    chart = go.Figure(go.Bar(x=['Unsteered', 'Full Δ · scale 2', 'Rank-1 Δ · scale 4'], y=[100*rate(r) for r in (baseline, full, rank1)], marker_color=['#9ca3af','#166b78','#c38434'], text=[f'{round(rate(r)*len(r))}/{len(r)}' for r in (baseline,full,rank1)], textposition='outside'))
    chart.update_layout(title='Confirmation on 20 fresh English questions', yaxis_title='Predominantly Cyrillic answers (%)', yaxis_range=[0,110], height=380, template='plotly_white', margin=dict(t=70,b=65,l=65,r=25))
    heat = make_subplots(rows=1, cols=2, subplot_titles=['Full empirical delta','Rank-1 empirical delta'])
    labels = [f'{g} · '+('once' if p == 0 else f'every {p} token(s)') for g in ('all','early','late') for p in (0,1,4)]
    for col, rank in enumerate((0,1),1):
        folder = 'matched/full' if rank == 0 else 'matched/rank1'
        z = [[100*rate(read(folder,f'r{rank}_{g}_p{p}_s{s}')) for s in ('0.5','1','2','4')] for g in ('all','early','late') for p in (0,1,4)]
        heat.add_trace(go.Heatmap(x=['0.5','1','2','4'],y=labels,z=z,zmin=0,zmax=100,colorscale='Viridis',showscale=col==2,colorbar=dict(title='RU %'),hovertemplate='%{y}<br>Scale %{x}<br>%{z}%<extra></extra>'),row=1,col=col)
    heat.update_layout(height=560,template='plotly_white',margin=dict(t=70,b=55,l=150,r=60))
    heat.update_xaxes(title='Scale'); heat.update_yaxes(autorange='reversed')
    figs = [to_html(f,full_html=False,include_plotlyjs='inline' if i==0 else False,config={'responsive':True,'displaylogo':False}) for i,f in enumerate((chart,heat))]
    examples = []
    for question in ('What is an algorithm?', 'Why do we see lightning before hearing thunder?', "What is the purpose of a computer's memory?"):
        cells = []
        for label,rows in [('Baseline',baseline),('Full delta',full),('Rank-1 delta',rank1)]:
            row = next(r for r in rows if r['question']==question)
            cells.append(f'<div><h4>{label} <small>{row["detected_language"]} · {row["language_confidence"]:.0%} confidence</small></h4><blockquote>{escape(row["response"])}</blockquote></div>')
        examples.append(f'<section class="example"><h3>{escape(question)}</h3>'+''.join(cells)+'</section>')
    inventory = []
    for path in sorted((DATA/'matched').glob('*/*.jsonl')):
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        inventory.append(f'<tr><td>{path.stem}</td><td>{len(rows)}</td><td>{sum(r["detected_language"]=="ru" for r in rows)}</td><td>{sum(r["detected_language"]=="mixed" for r in rows)}</td></tr>')
    assert len(inventory) == 74, f'Expected 74 sweep artifacts, found {len(inventory)}'
    article = r'''
<header><p class="eyebrow">EXPERIMENTAL NOTEBOOK · QWEN3.5-9B · 10 SEPTEMBER 2026</p>
<h1>From translation-state geometry to Russian answer steering in GDN</h1>
<p class="dek">A reconstruction of the experiments, implementation failures, and evidence that led to a working recurrent-state intervention.</p></header>
<aside><strong>Main result.</strong> With eight matched bilingual calibration answers, a single additive intervention in GDN memory produced predominantly Russian-script answers to 18 of 20 fresh English questions. A rank-1 approximation per head produced 17 of 20. Both unsteered confirmation runs produced 0 of 20. This is encouraging evidence of language control; it is not yet a comprehensive quality or robustness evaluation.</aside>
<p>This article summarizes the documented hypotheses, engineering decisions, and observed results. It reconstructs the experimental rationale from source code, saved artifacts, and the recorded run history. Early interpretations that the evidence later contradicted are explicitly corrected.</p>
<nav><a href="#geometry">1. Initial geometry</a> · <a href="#clamp">2. Clamping</a> · <a href="#debug">3. Invalid experiments</a> · <a href="#pivot">4. Strategy change</a> · <a href="#method">5. Working method</a> · <a href="#results">6. Results</a> · <a href="#limits">7. Limits</a> · <a href="#interpretability">8. Interpretability</a> · <a href="#next">9. Alternatives</a></nav>

<h2 id="geometry">1. The starting question: what differs between translated texts?</h2>
<p>The initial request was to collect recurrent GDN states for matching texts in two languages, compare their first singular vectors, and plot head-by-layer similarities. The collection used Qwen/Qwen3.5-9B and 200 English–Russian pairs streamed from OPUS-100, with both texts strictly longer than 20 whitespace-separated words. On airi.gpu3, physical GPU 2 processed batches of 32 pairs, or 64 texts per forward, in bfloat16. A second forward pass computed each pair's alignment with the final mean difference. The artifact covers 24 GDN layers and 32 heads per layer.</p>
<p>For this collector the sign convention was English minus Russian:</p>
\[D_i=S_i^{EN}-S_i^{RU},\qquad \bar D=\frac1N\sum_iD_i.\]
<p>It measured Frobenius norms, effective rank, and the absolute cosine between corresponding leading singular vectors. Absolute cosine removes the arbitrary sign of an SVD vector:</p>
\[S=U\Sigma V^\top,\qquad \operatorname{sim}_{k}=|u_{1,EN}^{\top}u_{1,RU}|,\qquad \operatorname{sim}_{v}=|v_{1,EN}^{\top}v_{1,RU}|.\]
<p><strong>Convention matters.</strong> The inspected Transformers implementation stores a key×value matrix and reads it as \(q^\top S\). Therefore the left singular vector is on the key side and the right is on the value side. The observations document uses the transposed, value×key convention, with readout \(Sq\). Statements about “left” and “right” must be translated before using them in code. The final additive experiment does not depend on naming these factors.</p>
<p>The collector already computed leading-vector similarity statistics internally. The change exposed them as sim_k and sim_v and added an unmasked heatmap alongside the rank-filtered view. It did <em>not</em> persist all per-example singular vectors: the saved artifact contains their aggregate similarities, not a reusable bank of factor vectors. The >20-word rule applies to this passage collection; later short-answer calibration deliberately uses a different dataset.</p>
<p>The supplied observations suggested early-layer states are often nearly rank one, whereas later states commonly have effective ranks around 5–14. Near-rank-one translated states can have similar key directions but greater value-side differences. A mean difference can nevertheless be high-rank even when its constituent states are low-rank: their dominant directions vary across examples.</p>
<p>The candidate dashboard combined alignment, contrast magnitude, and low rank:</p>
\[A_{\ell h}=\mathbb E_i\cos_F(D_i,\bar D),\quad M_{\ell h}=\frac{\|\bar D\|_F}{\mathbb E_i[(\|S_i^{EN}\|_F+\|S_i^{RU}\|_F)/2]},\]
\[C_{\ell h}=\min\{F_A(A_{\ell h}),F_M(M_{\ell h}),1-F_R(r_{\rm eff}(\bar D_{\ell h}))\}.\]
<p>Here \(F\) is the empirical fraction of all layer/head cells less than or equal to the argument. Effective rank is \(\exp[-\sum_jp_j\log p_j]\), with \(p_j=\sigma_j/\sum_k\sigma_k\). A bright candidate is jointly high on the first two criteria and low in rank. This is a heuristic ranking, not an estimated probability of successful steering. High alignment for layers 29/30, head 11, and low-rank candidates around layer 26 motivated targeted tests.</p>
<p>Several dashboard layout edits were made after screenshots showed squeezed heatmaps. These were presentation changes, not new scientific results. The original layout reused a long figure title as a colorbar title, so the earlier attribution of all layout problems to axis constraints was not established by a browser-based validation. This article uses short colorbar titles and responsive plots.</p>

<h2 id="clamp">2. Why try absolute coordinate clamping?</h2>
<p>The proposed procedure in steering_new.md asked whether an absolute memory coordinate would be more consistent across prompts than an additive displacement. With a unit-Frobenius direction \(\hat D\), define:</p>
\[z(S)=\langle S,\hat D\rangle_F,\qquad S'=S+(\lambda-z(S))\hat D.\]
<p>This makes \(\langle S',\hat D\rangle_F=\lambda\) and preserves the orthogonal component. A natural target is the Russian mean projection \(\lambda_{RU}=\langle\mathbb E[S^{RU}],\hat D\rangle_F\). The experimental generator implemented a strength parameter \(f\):</p>
\[S'=S+f(\lambda_{RU}-z(S))\hat D.\]
<p>At \(f=1\) this is the exact clamp; at \(f=0.5\) it moves halfway from the current prompt coordinate to the Russian target. This is <em>not</em> the same as interpolating between fixed empirical English and Russian coordinates. The initial calibration was only ten translation pairs and saved natural mean states. It did not estimate the per-direction EN/RU coordinate variances proposed in the specification.</p>
<p>A separate generate_clamp.py preserved the original generator. It implemented normalized full or truncated directions, additive controls, optional layers/heads, and post-prefill interventions. The additive control used unit-norm directions independently for each head, so its scale was a Frobenius displacement per head. That convention differs materially from the later empirical-delta scale.</p>

<h2 id="debug">3. The debugging episode: why many “negative” results were not experiments</h2>
<p>The first all-head screen compared a half-strength clamp with normalized addition on ten SQuAD questions. Later requests prompted specific layer/head tests and rank-1 additions, with scales progressively increased. There were also operational interruptions: uv was absent on the first host, a remote branch tracked a deleted upstream, GPU 2 was occupied by another process, and rank-1 reconstruction initially had an indexing error. Failed launches produced no usable results.</p>
<div class="warning"><strong>The decisive bug.</strong> The function argument named <code>heads</code> was overwritten by <code>for layer, heads in artifact['layers'].items()</code>. The mask then tested integer indices against a list of head dictionaries. Every index was treated as excluded, so every steering matrix was zero. This affected experiments after head masking was introduced, including nominal all-head runs.</div>
<p>Exactly zero before/after coordinates and unchanged answers were initially misinterpreted as evidence that the direction did not matter. That explanation was wrong. The first half-strength experiments preceded this masking change and were distinct initial negative screens; the subsequent masked runs cannot support conclusions about localization, scale, or rank. Merely increasing scale could not repair a zero direction.</p>
<p>Commit b19bb10 fixed the shadowing and rejected an all-zero selection. Focused checks verified unit norms on selected heads, zero norms on excluded heads, correct rank-1 construction, and the expected post-clamp coordinates. After this correction, scale-1 rank-1 addition changed answers and introduced the word “группой,” but also severe repetition. A corrected full-state clamp changed text while inspected answers remained English. This established that the cache mutation could affect generation, without establishing useful language steering.</p>
<p>The practical lesson was to validate the intervention itself before interpreting behavioral invariance: measure nonzero direction norms, actual state displacement, expected coordinate changes, and ideally the next-token logits. Early runs were not monitored continuously after every handoff; later sweeps were run to completion and their artifacts retrieved. The initial reports and conversational conclusions should not be treated as a clean preregistered experiment log.</p>

<h2 id="pivot">4. What motivated the strategy change?</h2>
<p>The observations made low-rank structure worth testing, but the stronger change was to align the calibration with the task. Bare translated passages probe memory while <em>reading</em> different languages. Evaluation asked an English chat prompt to <em>answer</em> in Russian. A passage contrast can contain differences in content distribution, tokenization, length, and reading state that are poorly matched to the desired output behavior.</p>
<p>We therefore collected English and Russian answer states behind the same English question and chat template. This still does not isolate language perfectly, but it holds the prompt and answer meaning much more directly matched. A second issue was timing: the old decoder had already computed its first-answer logits before modifying memory. Its first token therefore could not change. The new runner splits prefill before the final prompt token, applies the intervention, and processes that last token through the altered cache.</p>
<p>Finally, independent unit normalization can amplify tiny, noisy contrasts in weak heads. The new additive experiment retains the measured singular values and head magnitudes. Repeated application was added as a test of whether recurrence rapidly erases a one-time perturbation. These changes were made together, so the resulting improvement does not identify which change was individually necessary.</p>

<h2 id="method">5. The working method</h2>
<p>Eight manually paired English/Russian answers calibrated the direction. Topics included the sky, plants, birds, sleep, volcanoes, ice, libraries, and washing hands. Both language versions followed exactly the same English prompt, rendered with the model's chat template and thinking disabled. The answers were teacher-forced text appended to the assistant prefix; no Russian instruction was supplied to evaluation prompts.</p>
\[\mu_{\ell h}^{L}=\frac18\sum_{i=1}^8S_{\ell h}(P(q_i)\Vert a_i^L),\qquad \Delta_{\ell h}=\mu_{\ell h}^{RU}-\mu_{\ell h}^{EN}.\]
<p>This reverses the original collector's sign convention so that a positive additive scale points toward Russian. Full empirical deltas were compared to truncated SVD deltas:</p>
\[\Delta=U\Sigma V^\top,\qquad \Delta_r=\sum_{j=1}^r\sigma_ju_jv_j^\top.\]
<p>Rank one means the best Frobenius rank-1 approximation <em>separately at every head</em>. It is neither a single model-wide direction nor a rank-1 approximation of the current prompt state. The unnormalized leading singular value remains in the update.</p>
\[S_{\ell h}\leftarrow S_{\ell h}+\alpha\Delta_{\ell h}^{(r)}.\]
<p>The successful configuration applies this once across all 24 GDN layers, just before the final prompt token. The subsequent model forward both evolves that memory and computes the first answer logits. The experiment changes recurrent memory; it does not directly overwrite the attention KV cache or the convolution cache.</p>
<p>The screen evaluated ten separate everyday questions. It crossed two delta ranks (full and rank one), three layer groups (all, early layers &lt;12, late layers ≥24), three schedules (once, every token, every four tokens), and scales 0.5, 1, 2, and 4. Two zero-scale baselines give 74 conditions and 740 answers. Greedy decoding used a 64-token cap and a batch of ten; model weights were loaded once per rank sweep. Runs used GPUs 0/1 on airi.gpu after earlier exploration on airi.gpu3.</p>
<p>Confirmation used twenty new questions, fixed full-delta scale 2 and rank-1 scale 4, one-time/all-layer steering, and corresponding zero-scale controls. This produced another 80 answers in batches of twenty with a 128-token cap. Both positive settings reused the same eight calibration pairs. These confirmation questions were distinct from the calibration and the parameter-screen questions.</p>

<h2 id="results">6. Results and examples</h2>
<p>Language labels are a script heuristic: the fraction of alphabetic characters in the Cyrillic Unicode range U+0400–U+052F. “ru” means at least 80%; “mixed” means 20–80%; below 20% is “other.” Thus Russian percentages below are not independent language-model judgments or human correctness scores. Earlier generator labels used a looser 20% threshold and are not directly comparable.</p>
@@CONFIRM@@
<p>The confirmation rates were 90% (18/20) for full deltas and 85% (17/20) for rank-1 deltas, against 0/20 for both controls. A one-answer difference does not establish superiority. The scales also differ, so this is not an equal-strength comparison of rank choices. Full-delta failures included the computer-memory question and the compass question, which remained English.</p>
@@HEAT@@
<p>The heatmap shows the entire screen, not only successful selections. One-time all-layer steering reached 9/10 at full scale 2 and rank-1 scale 4. Every early-only setting produced zero predominantly Cyrillic answers, although a few mixed outputs occurred. One-time late-only steering also produced zero predominantly Cyrillic answers. Repeated late-only steering could induce Cyrillic or mixed output, but that does not establish comparable quality. A rank-1 all-layer update every four tokens at scale 4 reached 10/10 on script detection; it was not the setting carried into confirmation. That condition warrants a quality review instead of being silently ignored as a potentially useful alternative.</p>
<p>Below are saved confirmation outputs, reproduced without rewriting. They illustrate coherent transfer, and a setting that can remain English. Generation caps can truncate an otherwise useful answer; no EOS-based truncation flag was saved.</p>
@@EXAMPLES@@
<p>Manual inspection found relevant Russian explanations and instructions rather than just isolated Cyrillic words. It also found errors: some answers about flowers and onion chemistry are questionable, and Russian grammar or spelling is occasionally awkward. The experiments did not systematically compare factual accuracy against the baseline, so these mistakes cannot all be attributed to steering.</p>

<h2 id="limits">7. Why this is promising—and what is still unproven</h2>
<p>The strongest evidence is an intervention/control comparison on fresh English questions: identical prompt construction and deterministic decoding, with only recurrent-state addition changing. No Russian instruction or answer prefix is inserted during evaluation. Transfer across new subject matter makes simple copying of calibration answers implausible. The rank-1 result shows that retaining all singular modes of each mean difference is unnecessary for a strong effect in this setup.</p>
<p>However, eight calibration pairs and twenty confirmation questions are small and hand-selected. The model already knows Russian; this steers an existing capability. The experiment neither trains a new capability nor demonstrates generalization to other models, languages, long conversations, or adversarial prompts. Scale 4 is an extrapolation beyond the mean displacement, not a guaranteed natural-state-range intervention.</p>
<p>Calibration used final states after complete short answers, whereas intervention happens at the generation boundary. The setup reduces contextual mismatch but does not remove it. Calibration languages have different token lengths and can differ in lexical or formatting patterns. Padding and batch-size invariance deserve an explicit check. The stored relative-intervention statistic averages Frobenius displacement/state-norm ratios across heads, examples, layers, and applications; it is not a maximum, per-example quality measure, or rank measurement.</p>
<p>Most importantly, matched calibration, first-token timing, empirical scaling, and the evaluation questions changed together. To claim that one of them explains the gain requires controlled ablations. We have a working combined configuration, not a causal decomposition of its success.</p>

<h2 id="interpretability">8. What this tells us about a GDN layer</h2>
<p>GDN recurrent memory is a causal handle on answer language. Modifying a relatively simple contrast in that memory changes downstream generation, showing that state geometry can support more than descriptive similarity plots. Language-related information is accessible to later readout and decoding through the recurrent state.</p>
<p>The rank-1-per-head result is consistent with compact local components participating in that control. It does not imply that the full state is rank one, that language occupies one global dimension, or that each selected head is necessary. All-layer intervention succeeds where the particular early-only and late-only one-time interventions do not. This is compatible with distributed control, but also with a crucial untested middle-layer subset or interactions between layers. Necessity and localization remain open.</p>
<p>A coherent translation contrast is not automatically a behavioral controller. Conversely, a controller need not provide a complete semantic explanation of a head. The next interpretability step is to connect the changed memory to actual readout, track propagation through layers and tokens, and test ablations with norm-matched controls.</p>

<h2 id="next">9. Alternative approaches and the experiments they would answer</h2>
<table><thead><tr><th>Approach</th><th>Motivation and decisive comparison</th><th>Status</th></tr></thead><tbody>
<tr><td>Factor-only value steering</td><td>Preserve the current key direction and change the value factor. In key×value storage, use an update like \(k\,\delta v^\top\). Align SVD signs and verify correspondence to actual value vectors before averaging. Compare against norm-matched full and rank-1 deltas.</td><td>Proposed, not run</td></tr>
<tr><td>Matched coordinate clamping</td><td>Reuse the successful answer calibration, estimate coordinate means and variances, and compare one-time versus repeated mild clamping. This tests whether a stable absolute coordinate avoids additive accumulation.</td><td>Old passage clamps tested; matched/repeated clamps not run</td></tr>
<tr><td>Timing ablation</td><td>Hold the learned delta fixed and compare intervention before the final prompt token versus after full prefill. Is changing the first token critical?</td><td>Not run</td></tr>
<tr><td>Calibration/scaling ablation</td><td>Compare passage and answer contrasts on the same evaluation questions; compare raw and per-head normalized updates with matched total Frobenius norm.</td><td>Not run</td></tr>
<tr><td>Localization and subspace control</td><td>Test middle layers, leave-one-layer-out, sparse head subsets, and ranks 2/4. Learn orthonormal matrix directions and clamp their coordinates jointly.</td><td>Limited groups tested; systematic localization/subspace tests not run</td></tr>
<tr><td>Readout-aware steering</td><td>Optimize observable change \(q^\top\Delta S\) rather than matrix norm alone. Compare query-accessible directions with equal-norm random or orthogonal directions.</td><td>Proposed, not run</td></tr>
<tr><td>Other languages and quality controls</td><td>English→French and reverse steering would test language specificity. Larger independent prompts, native-language evaluation, and factual/repetition scoring would test robustness.</td><td>No French collection or steering completed</td></tr>
</tbody></table>

<h2>10. Reproducibility and evidence</h2>
<p>The active runner is <a href="../../experiments/steering/squad_sweep.py">squad_sweep.py</a>. It uses a saved translation-state delta, Qwen/Qwen3.5-9B, bfloat16, and greedy decoding; the project pins Torch 2.7.1 and Transformers 5.14.1. This narrative does not substitute those dependency declarations for a complete per-run environment snapshot.</p>
<pre>CUDA_VISIBLE_DEVICES=0 uv run python experiments/steering/squad_sweep.py \
  --deltas outputs/en_ru_500.pt --output outputs/squad_en_ru --questions 100</pre>
<p>Use new output directories to preserve existing results. Before comparing reruns, check model revision, dependencies, padding behavior, and numerical determinism.</p>
<p>Relevant commits: c221122 (translation similarity metrics), 29e61fe (experimental clamps), 1a87048 (head selection, containing the later-discovered shadowing bug), 1de1f3d (rank reconstruction fix), b19bb10 (zero-direction bug fix), 5c34360 (matched sweep), b0ff6bd (confirmation and reporting).</p>
<p>Primary local evidence: <a href="language_states_en_ru_200.html">200-pair state dashboard</a>; <a href="matched/report.html">full sweep examples</a>; <a href="confirmation/report.html">confirmation examples</a>; <a href="../../observations.md">observations.md</a>; <a href="../../steering_new.md">steering_new.md</a>. The observations document supplies geometric findings, whereas the new JSONL artifacts supply the behavioral evidence. The key×value convention was checked against the installed Transformers Qwen3.5 recurrent update and readout implementation.</p>
<details><summary>All 74 screening conditions: artifact inventory</summary><table><tr><th>Condition</th><th>Answers</th><th>ru</th><th>mixed</th></tr>@@INVENTORY@@</table></details>
<footer>Research status: demonstrated EN→RU steering on a small fresh-question evaluation. Broader quality validation, causal ablations, factor-only steering, and matched repeated clamping remain open.</footer>
'''
    article = article.replace('@@CONFIRM@@',figs[0]).replace('@@HEAT@@',figs[1]).replace('@@EXAMPLES@@',''.join(examples)).replace('@@INVENTORY@@',''.join(inventory))
    shell = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>From translation-state geometry to GDN steering</title>
<script>window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},svg:{fontCache:'global'}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>:root{color-scheme:light}body{margin:0;background:#f7f5f0;color:#242c32;font:18px/1.7 Georgia,serif}article{max-width:1050px;margin:auto;padding:65px 32px;background:white}h1,h2,h3,h4,nav,summary,th,.eyebrow{font-family:system-ui,sans-serif}h1{font-size:clamp(32px,5vw,52px);line-height:1.15;letter-spacing:-1.5px}h2{font-size:28px;margin-top:56px;border-top:1px solid #ddd;padding-top:24px}h3{font-size:21px}h4{font-size:16px;margin-bottom:5px}small{font-weight:normal;color:#64748b;display:block}.eyebrow{font-size:12px;letter-spacing:2px;color:#166b78}.dek{font-size:23px;color:#52606a}nav{font-size:14px;margin:30px 0}a{color:#166b78}aside,.warning{padding:20px 25px;border-left:4px solid #166b78;background:#edf6f5}.warning{border-color:#c38434;background:#fff6e8}table{font:14px/1.5 system-ui,sans-serif;border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#eef3f4}pre{font-size:13px;overflow:auto;padding:20px;background:#eef3f4}blockquote{margin:10px 0 22px;padding:14px 20px;background:#f7f8fa;white-space:pre-wrap;font-size:16px}.example{border:1px solid #ddd;padding:20px;margin:25px 0}details{margin:25px 0}summary{cursor:pointer}footer{font-size:15px;color:#52606a;border-top:1px solid #ccc;padding-top:25px;margin-top:45px}mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;padding:8px 0}@media(max-width:700px){article{padding:25px 15px}body{font-size:16px}table{font-size:12px}}@media print{body{background:white}article{max-width:none;padding:0}nav{display:none}details{display:block}}</style></head><body><article>'''
    target=DATA/'steering_article.html'
    target.write_text(shell+article+'</article></body></html>')
    print(target)


if __name__ == '__main__': main()
