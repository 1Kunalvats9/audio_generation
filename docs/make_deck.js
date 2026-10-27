const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/Users/kunalvats/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/1b99554b-6c54-4954-9f13-2f42b5e3278f/434e0e0c-5bb9-4a58-869f-78c1d729311d/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const REPO = "/Users/kunalvats/Downloads/Coding/AML_project";
const OUT = path.join(REPO, "docs/Phonetic_Guided_Speech_Generation.pptx");

// ---- read the real results -------------------------------------------------
function readCsv(p) {
  const lines = fs.readFileSync(p, "utf8").trim().split("\n");
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    // results CSVs here have no quoted commas in the columns we read
    const cells = l.split(",");
    return Object.fromEntries(head.map((h, i) => [h.trim(), (cells[i] || "").trim()]));
  });
}

const dur = readCsv(path.join(REPO, "outputs/results/duration_models.csv"));
const test = dur.filter((r) => r.split === "test");
const maeOf = (m) => Number(test.find((r) => r.model === m).mae_ms);
const framesOf = (m) => Number(test.find((r) => r.model === m).mae_frames);
const improvement = ((1 - framesOf("gbt") / framesOf("mean")) * 100).toFixed(1);

const wer = {};
for (const s of ["R0", "R1", "A1", "A2"]) {
  const f = path.join(REPO, `outputs/results/wer_${s}.csv`);
  if (!fs.existsSync(f)) continue;
  const rows = readCsv(f);
  const mean = (k) => rows.reduce((a, r) => a + Number(r[k]), 0) / rows.length;
  wer[s] = { n: rows.length, wer: mean("wer") * 100, cer: mean("cer") * 100 };
}
console.log("duration MAE (test, ms):", { mean: maeOf("mean"), ridge: maeOf("ridge"), gbt: maeOf("gbt") });
console.log("WER:", wer);
const f2 = (x) => x.toFixed(2);
const pc = (s) => (wer[s] ? f2(wer[s].wer) + "%" : "n/a");
const cc = (s) => (wer[s] ? f2(wer[s].cer) + "%" : "n/a");

// ---- theme -----------------------------------------------------------------
const THEME = {
  name: "Phonetic Speech",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16213E", lt1: "FFFFFF",
    dk2: "065A82", lt2: "EEF4F7",
    accent1: "1C7293", accent2: "02C39A", accent3: "065A82",
    accent4: "21295C", accent5: "97BC62", accent6: "B85042",
    hlink: "1C7293", folHlink: "6D2E46",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 — must precede addSlide
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Kunal Vats, Tanishk Agrawal, Pushkar Jain, Shaurya Sharma, Chaitanya Shekhawat";
pres.title = "Phonetic-Guided Speech Generation";
const C = pres.SchemeColor;

// ---- layouts ---------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.5, w: 11.7, h: 1.5,
        fontSize: 42, bold: true, color: C.background1, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.0, w: 11.7, h: 0.8,
        fontSize: 18, color: "7FB5C9", align: "left" }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT_LIGHT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 11.9, h: 0.78,
        fontSize: 36, bold: true, color: C.accent3, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "9AA7B4", align: "right" },
});

pres.defineSlideMaster({
  title: "CONTENT_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 11.9, h: 0.78,
        fontSize: 36, bold: true, color: C.background1, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "6E8A99", align: "right" },
});

// ============================ SLIDE 1 — title + team =======================
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Phonetic-Guided Speech Generation", { placeholder: "title" });
s.addText("How much does timing matter in synthetic speech?", { placeholder: "body" });


const team = [
  ["Kunal Vats", "Data — cleaning, splits, shared settings  ·  Duration model (with Pushkar)"],
  ["Pushkar Jain", "Duration model — features, 3 models, error analysis (with Kunal)"],
  ["Tanishk Agrawal", "Alignment — forced alignment, phone durations"],
  ["Shaurya Sharma", "Sound — spectrograms, templates, vocoder, synthesis"],
  ["Chaitanya Shekhawat", "Evaluation — automatic listening test, results"],
];
s.addTable(
  team.map(([n, r]) => [
    { text: n, options: { bold: true, color: "FFFFFF", fontSize: 13 } },
    { text: r, options: { color: "BFD4DE", fontSize: 12 } },
  ]),
  { x: 0.8, y: 4.05, w: 11.7, colW: [3.2, 8.5], rowH: 0.42,
    border: { type: "solid", color: "2A3C56", pt: 1 }, margin: 0.08, objectName: "teamTable" }
);
s.addText("Advanced Machine Learning  ·  LJSpeech 1.1 (13,100 clips, 23.9 h, one speaker)  ·  Built on FastSpeech 2 (ICLR 2021)",
  { x: 0.8, y: 6.5, w: 11.7, h: 0.4, fontSize: 11, color: "7FB5C9", isTextBox: true, margin: 0 });
s.addNotes("We built a small text-to-speech system, but the goal was not the best-sounding voice. It was to answer one measurable question, on the next slide.");

// ============================ SLIDE 2 — the question =======================
pres.addSection({ title: "The question" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The question" });
s.addText("Speech synthesis has to get two separate things right", { placeholder: "title" });

function card(slide, x, y, w, h, name) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06,
    fill: { color: C.background2 }, line: { color: "D7E3EA", width: 1 },
    shadow: { type: "outer", color: "9AB2BF", blur: 8, offset: 2, angle: 90, opacity: 0.25 },
    objectName: name });
}

card(s, 0.7, 1.6, 5.7, 2.0, "cardTiming");
s.addShape(pres.ShapeType.ellipse, { x: 1.0, y: 1.9, w: 0.5, h: 0.5, fill: { color: C.accent1 }, objectName: "dot1" });
s.addText("1", { x: 1.0, y: 1.9, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("TIMING — how long does each sound last?", { x: 1.65, y: 1.92, w: 4.5, h: 0.45, fontSize: 15, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText('In "cat", the sounds K · AE · T are not equal. The vowel is longer. The same sound stretches at the end of a sentence and shortens in a fast phrase.',
  { x: 1.0, y: 2.55, w: 5.1, h: 0.9, fontSize: 13, color: C.text2, isTextBox: true, margin: 0 });

card(s, 6.9, 1.6, 5.7, 2.0, "cardSound");
s.addShape(pres.ShapeType.ellipse, { x: 7.2, y: 1.9, w: 0.5, h: 0.5, fill: { color: C.accent2 }, objectName: "dot2" });
s.addText("2", { x: 7.2, y: 1.9, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("SOUND — what does each sound sound like?", { x: 7.85, y: 1.92, w: 4.5, h: 0.45, fontSize: 15, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText("The hiss of S, the ring of AE. The frequency content that makes each phone recognisable.",
  { x: 7.2, y: 2.55, w: 5.1, h: 0.9, fontSize: 13, color: C.text2, isTextBox: true, margin: 0 });

s.addText("The problem: modern systems learn both at the same time, inside one neural network. So when the output sounds wrong, you cannot tell which half is to blame.",
  { x: 0.7, y: 3.85, w: 11.9, h: 0.6, fontSize: 15, color: C.accent6, bold: true, isTextBox: true, margin: 0 });

s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.65, w: 11.9, h: 2.0, rectRadius: 0.05,
  fill: { color: C.accent3 }, objectName: "questionBox" });
s.addText("Our question", { x: 1.1, y: 4.85, w: 5, h: 0.4, fontSize: 13, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText("If we keep the “sound” half completely fixed and change only where the timing comes from, how much worse does the speech get?",
  { x: 1.1, y: 5.2, w: 11.1, h: 0.6, fontSize: 19, bold: true, color: "FFFFFF", italic: true, isTextBox: true, margin: 0 });
s.addText([
  { text: "A1", options: { bold: true, color: C.accent2, fontSize: 14 } },
  { text: " = rebuild using the real, measured durations        ", options: { color: "D8E7EE", fontSize: 14 } },
  { text: "A2", options: { bold: true, color: C.accent2, fontSize: 14 } },
  { text: " = rebuild using durations predicted by our model", options: { color: "D8E7EE", fontSize: 14 } },
], { x: 1.1, y: 5.95, w: 11.1, h: 0.5, isTextBox: true, margin: 0 });
s.addNotes("This is the whole project in one idea. If they ask only one question it will be this. Honest framing: we are not inventing a new model, we are measuring an assumption FastSpeech 2 makes but never isolates.");


// ============================ SLIDE 3 — what we did ========================
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Approach" });
s.addText("What we actually did", { placeholder: "title" });

s.addText("Five steps", { x: 0.7, y: 1.35, w: 6, h: 0.35, fontSize: 15, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addTable([
  [{ text: "Step", options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 11 } },
   { text: "What happens", options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 11 } },
   { text: "Output", options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 11 } }],
  ["1  Clean the text", "spell out numbers, strip punctuation", "0 digits left in 13,100 clips"],
  ["2  Align", "find where every sound starts and ends", "949,856 phones, 70 types"],
  ["3  Spectrograms", "turn audio into 80-band pictures", "1 frame = 11.61 ms"],
  ["4  Predict durations", "the machine-learning part", "905,833 training examples"],
  ["5  Rebuild audio", "average sound per phone + vocoder", "100 clips x 3 systems"],
].map((row, i) => i === 0 ? row : row.map((c, j) => ({
  text: c, options: { fontSize: 11, color: j === 0 ? "065A82" : "3A4A58", bold: j === 0 } }))),
  { x: 0.7, y: 1.75, w: 7.3, colW: [1.9, 3.1, 2.3], rowH: 0.34,
    border: { type: "solid", color: "D7E3EA", pt: 1 }, margin: 0.07, objectName: "stepsTable" });

s.addText("Is the alignment trustworthy? We checked with a control", { x: 0.7, y: 4.3, w: 7.3, h: 0.35, fontSize: 15, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addText([
  { text: "94.6%", options: { fontSize: 30, bold: true, color: "02C39A" } },
  { text: "  of sound boundaries land where the loudness changes", options: { fontSize: 13, color: "3A4A58" } },
], { x: 0.7, y: 4.7, w: 7.3, h: 0.55, isTextBox: true, margin: 0 });
s.addText([
  { text: "53.7%", options: { fontSize: 30, bold: true, color: "B85042" } },
  { text: "  same test, boundaries shifted on purpose (the control)", options: { fontSize: 13, color: "3A4A58" } },
], { x: 0.7, y: 5.3, w: 7.3, h: 0.55, isTextBox: true, margin: 0 });
s.addText("53.7% is what a broken aligner would score. 94.6% against that control is why everything downstream rests on something real.",
  { x: 0.7, y: 5.95, w: 7.3, h: 0.7, fontSize: 11, italic: true, color: "6B7A88", isTextBox: true, margin: 0 });

card(s, 8.4, 1.75, 4.2, 4.85, "mlCard");
s.addText("The machine-learning model", { x: 8.7, y: 1.98, w: 3.6, h: 0.4, fontSize: 14, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText([
  { text: "Task: given a phone and its context, predict its length", options: { bullet: true, breakLine: true } },
  { text: "21 features, all from the text — the phone, its neighbours, stress, position in word and sentence, and distance to the nearest pause", options: { bullet: true, breakLine: true } },
  { text: "3 models compared: per-phone average (baseline), ridge regression, gradient-boosted trees", options: { bullet: true, breakLine: true } },
  { text: "Trained on 12,501 clips, tuned on 300, reported on 299 clips from unseen books", options: { bullet: true } },
], { x: 8.7, y: 2.45, w: 3.6, h: 3.9, fontSize: 11.5, color: "3A4A58", paraSpaceAfter: 10, isTextBox: true, margin: 0 });
s.addNotes("The distance-to-nearest-pause feature captures a real effect: the sound just before a pause gets stretched. We did not invent it, we measured it. Mention the control test — it shows we validated rather than assumed.");

// ============================ SLIDE 4 — results ============================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Results" });
s.addText("Results", { placeholder: "title" });

// duration model chart (native)
s.addText("Duration error on unseen books (lower is better)", { x: 0.7, y: 1.32, w: 5.9, h: 0.35, fontSize: 14, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addChart(pres.ChartType.bar, [{
  name: "Duration MAE (ms)",
  labels: ["Per-phone average\n(baseline)", "Ridge regression", "Boosted trees\n(ours)"],
  values: [maeOf("mean"), maeOf("ridge"), maeOf("gbt")],
}], {
  x: 0.6, y: 1.74, w: 6.0, h: 2.55,
  barDir: "col", chartColors: ["9AB2BF", "1C7293", "02C39A"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0" ms"',
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: "3A4A58",
  catAxisLabelColor: "3A4A58", catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 10,
  valAxisLabelColor: "6B7A88", valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 9,
  valAxisTitle: "mean absolute error (ms)", showValAxisTitle: true,
  valAxisTitleColor: "6B7A88", valAxisTitleFontSize: 9, valAxisTitleFontFace: "+mn-lt",
  valGridLine: { color: "E6EDF1", size: 1 }, catGridLine: { style: "none" },
  showLegend: false, showTitle: false, barGapWidthPct: 60,
});
s.addText([
  { text: improvement + "% ", options: { fontSize: 24, bold: true, color: "02C39A" } },
  { text: "better than the baseline — our features earn their place", options: { fontSize: 13, color: "3A4A58" } },
], { x: 0.7, y: 4.35, w: 5.9, h: 0.5, isTextBox: true, margin: 0 });

card(s, 0.6, 5.0, 6.0, 1.65, "surpriseCard");
s.addText("The surprise", { x: 0.9, y: 5.15, w: 5.4, h: 0.3, fontSize: 11, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText("We expected A2 (predicted) to be worse than A1 (true durations). It was not \u2014 21.15% vs 23.65%. The difference is marginal (better on 40 of 100 clips, p = 0.05 to 0.10), so the honest claim is that duration error had no measurable effect on intelligibility.",
  { x: 0.9, y: 5.45, w: 5.4, h: 1.1, fontSize: 10.5, color: "3A4A58", isTextBox: true, margin: 0 });

// intelligibility table
s.addText("Can a machine still understand the speech?", { x: 7.0, y: 1.32, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 11 } });
s.addTable([
  [hdr("System"), hdr("What it is"), hdr("WER"), hdr("CER")],
  [{ text: "R0", options: { bold: true, fontSize: 11, color: "065A82" } }, { text: "real recording", options: { fontSize: 10.5, color: "3A4A58" } }, { text: pc("R0"), options: { fontSize: 11, bold: true, color: "3A4A58" } }, { text: cc("R0"), options: { fontSize: 10.5, color: "6B7A88" } }],
  [{ text: "R1", options: { bold: true, fontSize: 11, color: "065A82" } }, { text: "real spectrogram -> vocoder", options: { fontSize: 10.5, color: "3A4A58" } }, { text: pc("R1"), options: { fontSize: 11, bold: true, color: "3A4A58" } }, { text: cc("R1"), options: { fontSize: 10.5, color: "6B7A88" } }],
  [{ text: "A1", options: { bold: true, fontSize: 11, color: "065A82" } }, { text: "templates + TRUE durations", options: { fontSize: 10.5, color: "3A4A58" } }, { text: pc("A1"), options: { fontSize: 11, bold: true, color: "3A4A58" } }, { text: cc("A1"), options: { fontSize: 10.5, color: "6B7A88" } }],
  [{ text: "A2", options: { bold: true, fontSize: 11, color: "02C39A" } }, { text: "templates + PREDICTED durations", options: { fontSize: 10.5, color: "3A4A58" } }, { text: pc("A2"), options: { fontSize: 11, bold: true, color: "02C39A" } }, { text: cc("A2"), options: { fontSize: 10.5, color: "6B7A88" } }],
], { x: 7.0, y: 1.74, w: 5.6, colW: [0.8, 2.6, 1.1, 1.1], rowH: 0.36,
     border: { type: "solid", color: "D7E3EA", pt: 1 }, margin: 0.07, objectName: "werTable" });

s.addText("Where does the quality actually go?", { x: 7.0, y: 4.05, w: 5.6, h: 0.32, fontSize: 13, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
const chain = [
  ["R0 \u2192 R1", "+0.63 pp", "the vocoder \u2014 almost free", "1C7293"],
  ["R1 \u2192 A1", "+21.17 pp", "averaged templates \u2014 all the loss is here", "B85042"],
  ["A1 \u2192 A2", "\u22122.50 pp", "predicting durations \u2014 cost nothing", "02C39A"],
];
let cy = 4.45;
for (const [step, cost, what, col] of chain) {
  s.addText(step, { x: 7.0, y: cy, w: 1.15, h: 0.3, fontSize: 11, bold: true, color: "6B7A88", isTextBox: true, margin: 0 });
  s.addText(cost, { x: 8.1, y: cy, w: 1.2, h: 0.3, fontSize: 13, bold: true, color: col, isTextBox: true, margin: 0 });
  s.addText(what, { x: 9.35, y: cy + 0.02, w: 3.3, h: 0.3, fontSize: 10.5, color: "3A4A58", isTextBox: true, margin: 0 });
  cy += 0.42;
}
s.addText("The acoustic model matters about 30x more than duration accuracy.",
  { x: 7.0, y: 5.85, w: 5.6, h: 0.4, fontSize: 12, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addNotes("WER = Word Error Rate: we let Whisper listen and compare its transcript to the truth. It measures intelligibility, not naturalness. Our templates are one average spectrum per phone, so the audio is understandable but obviously synthetic - that is by design, so duration can be isolated.");

// ============================ SLIDE 5 — what we learned ====================
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Conclusion" });
s.addText("What we learned", { placeholder: "title" });

const findings = [
  ["Forced alignment gives genuinely reliable timing",
   "94.6% of boundaries verified against a 53.7% control, all 13,100 clips aligned, none excluded. It also reproduced known phonetics we never built in: stressed vowels 108 ms vs 67 ms."],
  ["Context features clearly beat a per-phone average",
   improvement + "% lower duration error (30.61 ms \u2192 19.65 ms). Phone identity with stress, the neighbouring phones, and distance to the nearest pause carry the gain."],
  ["We set out to measure one thing and learned the other matters ~30x more",
   "Predicting durations cost nothing measurable. The crude acoustic model cost 21 percentage points. So the next effort belongs in the acoustic model \u2014 exactly where a diffusion generator would sit."],
];
let y = 1.5;
for (const [h, d] of findings) {
  s.addShape(pres.ShapeType.ellipse, { x: 0.75, y: y + 0.04, w: 0.34, h: 0.34, fill: { color: C.accent2 }, objectName: "fdot" + y });
  s.addText(h, { x: 1.25, y: y, w: 11.3, h: 0.4, fontSize: 16, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText(d, { x: 1.25, y: y + 0.42, w: 11.3, h: 0.5, fontSize: 12.5, color: "AEC6D2", isTextBox: true, margin: 0 });
  y += 1.1;
}

s.addText("Being honest about the limits", { x: 0.75, y: 4.9, w: 11.5, h: 0.32, fontSize: 13, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText([
  { text: "A2 is told the clip's true total length, so it is not yet a fully autonomous system", options: { bullet: true, breakLine: true } },
  { text: "Word Error Rate measures intelligibility, not naturalness — we ran no human listening test", options: { bullet: true, breakLine: true } },
  { text: "One speaker only, so we cannot claim this generalises to other voices", options: { bullet: true } },
], { x: 0.75, y: 5.25, w: 11.5, h: 1.3, fontSize: 11.5, color: "8FA9B8", paraSpaceAfter: 6, isTextBox: true, margin: 0 });

s.addText("Next: predict total length too, replace templates with a learned acoustic model, run a listening test",
  { x: 0.75, y: 6.75, w: 11.5, h: 0.35, fontSize: 11, italic: true, color: "02C39A", isTextBox: true, margin: 0 });
s.addNotes("If pushed on why the synthetic audio is worse than the real recording: that is the expected consequence of a deliberately simple acoustic model. The project was never trying to win on audio quality - the experiment is the A1 vs A2 comparison, which is only meaningful because the acoustic model is held fixed.");

// ---- write ------------------------------------------------------------------
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
