const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/Users/kunalvats/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/1b99554b-6c54-4954-9f13-2f42b5e3278f/434e0e0c-5bb9-4a58-869f-78c1d729311d/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const REPO = "/Users/kunalvats/Downloads/Coding/AML_project";
const OUT = path.join(REPO, "docs/Phonetic_Guided_Speech_Generation.pptx");

// ---- read the real results so the slides can never drift from the data ------
function readCsv(p) {
  const lines = fs.readFileSync(p, "utf8").trim().split("\n");
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const cells = l.split(",");
    return Object.fromEntries(head.map((h, i) => [h.trim(), (cells[i] || "").trim()]));
  });
}
const R = (f) => path.join(REPO, "outputs/results", f);

const dur = readCsv(R("duration_models.csv"));
const test = dur.filter((r) => r.split === "test");
const val = dur.filter((r) => r.split === "val");
const pick = (rows, m, k) => Number(rows.find((r) => r.model === m)[k]);
const improvement = ((1 - pick(test, "gbt", "mae_frames") / pick(test, "mean", "mae_frames")) * 100).toFixed(1);

const wer = {};
for (const s of ["R0", "R1", "A1", "A2"]) {
  const rows = readCsv(R(`wer_${s}.csv`));
  const mean = (k) => rows.reduce((a, r) => a + Number(r[k]), 0) / rows.length;
  wer[s] = { wer: mean("wer") * 100, cer: mean("cer") * 100 };
}

const mosRows = readCsv(R("listening_study.csv"));
const mos = {};
for (const s of ["R0", "R1", "A1", "A2"]) {
  const v = mosRows.filter((r) => r.system === s).map((r) => Number(r.mean_mos));
  mos[s] = v.reduce((a, b) => a + b, 0) / v.length;
}

const imp = readCsv(R("feature_importance.csv"));
const impOf = (name) => Number((imp.find((r) => r[""] === name) || {}).importance || 0);

const f2 = (x) => x.toFixed(2);
console.log("duration MAE ms:", { mean: pick(test,"mean","mae_ms"), ridge: pick(test,"ridge","mae_ms"), gbt: pick(test,"gbt","mae_ms") });
console.log("WER:", Object.fromEntries(Object.entries(wer).map(([k,v]) => [k, f2(v.wer)])));
console.log("MOS:", Object.fromEntries(Object.entries(mos).map(([k,v]) => [k, f2(v)])));

// ---- theme ------------------------------------------------------------------
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
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Kunal Vats, Pushkar Jain, Tanishk Agrawal, Shaurya Sharma, Chaitanya Shekhawat";
pres.title = "Phonetic-Guided Speech Generation";
const C = pres.SchemeColor;

const INK = "3A4A58", MUTED = "6B7A88", LINE = "D7E3EA";

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
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.42, w: 11.9, h: 0.72,
        fontSize: 32, bold: true, color: C.accent3, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "9AA7B4", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.42, w: 11.9, h: 0.72,
        fontSize: 32, bold: true, color: C.background1, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "6E8A99", align: "right" },
});

// helpers
function card(slide, x, y, w, h, name, dark) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: dark ? "1E2A47" : C.background2 },
    line: { color: dark ? "2A3C56" : LINE, width: 1 }, objectName: name });
}
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 10.5 } });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 10.5, color: INK }, o) });

// ============================ 1 — title + team =============================
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Phonetic-Guided Speech Generation", { placeholder: "title" });
s.addText("How much does timing matter in synthetic speech?", { placeholder: "body" });
const team = [
  ["Kunal Vats", "Data — cleaning, splits, shared settings  ·  Duration model (with Pushkar)"],
  ["Pushkar Jain", "Duration model — features, 3 models, error analysis (with Kunal)"],
  ["Tanishk Agrawal", "Alignment — forced alignment, phone durations"],
  ["Shaurya Sharma", "Sound — spectrograms, templates, vocoder, synthesis"],
  ["Chaitanya Shekhawat", "Evaluation — WER harness, listening study, failure analysis"],
];
s.addTable(team.map(([n, r]) => [
  { text: n, options: { bold: true, color: "FFFFFF", fontSize: 13 } },
  { text: r, options: { color: "BFD4DE", fontSize: 12 } }]),
  { x: 0.8, y: 4.05, w: 11.7, colW: [3.2, 8.5], rowH: 0.42,
    border: { type: "solid", color: "2A3C56", pt: 1 }, margin: 0.08, objectName: "teamTable" });
s.addText("Advanced Machine Learning  ·  LJSpeech 1.1 (13,100 clips, 23.9 h, one speaker)  ·  Built on FastSpeech 2 (ICLR 2021)",
  { x: 0.8, y: 6.5, w: 11.7, h: 0.4, fontSize: 11, color: "7FB5C9", isTextBox: true, margin: 0 });
s.addNotes("We built a small text-to-speech system, but the goal was not the best-sounding voice. It was to answer one measurable question, on the next slide.");

// ============================ 2 — the question =============================
pres.addSection({ title: "The question" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The question" });
s.addText("Speech synthesis has to get two separate things right", { placeholder: "title" });

card(s, 0.7, 1.45, 5.7, 1.9, "cardTiming");
s.addShape(pres.ShapeType.ellipse, { x: 1.0, y: 1.72, w: 0.46, h: 0.46, fill: { color: C.accent1 }, objectName: "dot1" });
s.addText("1", { x: 1.0, y: 1.72, w: 0.46, h: 0.46, fontSize: 17, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("TIMING — how long does each sound last?", { x: 1.6, y: 1.74, w: 4.6, h: 0.42, fontSize: 14.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText('In "cat", the sounds K · AE · T are not equal. The vowel is longer. The same sound stretches before a pause and shortens in a fast phrase.',
  { x: 1.0, y: 2.3, w: 5.1, h: 0.9, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });

card(s, 6.9, 1.45, 5.7, 1.9, "cardSound");
s.addShape(pres.ShapeType.ellipse, { x: 7.2, y: 1.72, w: 0.46, h: 0.46, fill: { color: C.accent2 }, objectName: "dot2" });
s.addText("2", { x: 7.2, y: 1.72, w: 0.46, h: 0.46, fontSize: 17, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("SOUND — what does each sound sound like?", { x: 7.8, y: 1.74, w: 4.6, h: 0.42, fontSize: 14.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText("The hiss of S, the ring of AE. The frequency content that makes each phone recognisable.",
  { x: 7.2, y: 2.3, w: 5.1, h: 0.9, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });

s.addText("Modern systems learn both at once inside one neural network. So when the output is wrong, you cannot tell which half is to blame.",
  { x: 0.7, y: 3.55, w: 11.9, h: 0.5, fontSize: 14, color: C.accent6, bold: true, isTextBox: true, margin: 0 });

s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.25, w: 11.9, h: 2.4, rectRadius: 0.05, fill: { color: C.accent3 }, objectName: "questionBox" });
s.addText("Our question", { x: 1.1, y: 4.45, w: 5, h: 0.35, fontSize: 12.5, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText("Keep the “sound” half completely fixed, change only where the timing comes from. How much worse does the speech get?",
  { x: 1.1, y: 4.78, w: 11.1, h: 0.6, fontSize: 18, bold: true, color: "FFFFFF", italic: true, isTextBox: true, margin: 0 });
s.addText([
  { text: "A1", options: { bold: true, color: C.accent2, fontSize: 13.5 } },
  { text: "  templates + REAL measured durations", options: { color: "D8E7EE", fontSize: 13.5 } },
], { x: 1.1, y: 5.55, w: 5.3, h: 0.4, isTextBox: true, margin: 0 });
s.addText([
  { text: "A2", options: { bold: true, color: C.accent2, fontSize: 13.5 } },
  { text: "  templates + PREDICTED durations", options: { color: "D8E7EE", fontSize: 13.5 } },
], { x: 6.7, y: 5.55, w: 5.3, h: 0.4, isTextBox: true, margin: 0 });
s.addText("Everything else is identical, so the gap between A1 and A2 is caused by duration error and nothing else.",
  { x: 1.1, y: 6.0, w: 11.1, h: 0.4, fontSize: 12, color: "AECBD8", isTextBox: true, margin: 0 });
s.addNotes("This is the whole project in one idea. If they ask only one question it will be this. Honest framing: we are not inventing a new model, we are measuring an assumption FastSpeech 2 makes but never isolates.");

// ============================ 3 — the pipeline =============================
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Approach" });
s.addText("The pipeline: five steps", { placeholder: "title" });

const steps = [
  ["1", "Clean the text", "spell out numbers, strip punctuation", "0 digits left in 13,100 clips"],
  ["2", "Align", "find where every sound starts and ends", "949,856 phones, 70 types"],
  ["3", "Spectrograms", "turn audio into 80-band pictures", "1 frame = 11.61 ms"],
  ["4", "Predict durations", "the machine-learning part", "905,833 training examples"],
  ["5", "Rebuild audio", "phoneme templates + vocoder", "100 clips x 3 systems"],
];
let py = 1.3;
for (const [n, what, how, out] of steps) {
  s.addShape(pres.ShapeType.ellipse, { x: 0.72, y: py + 0.02, w: 0.32, h: 0.32, fill: { color: C.accent1 }, objectName: "st" + n });
  s.addText(n, { x: 0.72, y: py + 0.02, w: 0.32, h: 0.32, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText(what, { x: 1.18, y: py, w: 2.0, h: 0.34, fontSize: 12.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(how, { x: 3.2, y: py + 0.03, w: 2.6, h: 0.32, fontSize: 11, color: INK, isTextBox: true, margin: 0 });
  s.addText(out, { x: 5.85, y: py + 0.03, w: 2.3, h: 0.32, fontSize: 11, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
  py += 0.46;
}

s.addText("Is the alignment trustworthy? We tested it against a control", { x: 0.72, y: 3.8, w: 7.4, h: 0.35, fontSize: 14, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addText([{ text: "94.6%", options: { fontSize: 28, bold: true, color: "02C39A" } },
           { text: "   of phone boundaries land where the loudness changes", options: { fontSize: 12.5, color: INK } }],
  { x: 0.72, y: 4.2, w: 7.4, h: 0.5, isTextBox: true, margin: 0 });
s.addText([{ text: "53.7%", options: { fontSize: 28, bold: true, color: "B85042" } },
           { text: "   same test, boundaries shifted 5 frames on purpose", options: { fontSize: 12.5, color: INK } }],
  { x: 0.72, y: 4.78, w: 7.4, h: 0.5, isTextBox: true, margin: 0 });
s.addText("53.7% is what a broken aligner scores. The gap is why everything downstream rests on something real.\nIt also rediscovered a known fact we never encoded: stressed vowels 108 ms vs 67 ms unstressed.",
  { x: 0.72, y: 5.35, w: 7.4, h: 0.8, fontSize: 11, italic: true, color: MUTED, isTextBox: true, margin: 0 });

card(s, 8.5, 1.3, 4.1, 4.85, "dataCard");
s.addText("The data", { x: 8.8, y: 1.5, w: 3.5, h: 0.34, fontSize: 13.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addTable([
  [cell("LJSpeech 1.1", { bold: true }), cell("13,100 clips")],
  [cell("Speaker"), cell("1 female, 23.9 h")],
  [cell("Audio"), cell("22,050 Hz mono")],
  [cell("Clip length"), cell("1.1 – 10.1 s")],
  [cell("Train"), cell("12,501 clips", { bold: true })],
  [cell("Validation"), cell("300 clips", { bold: true })],
  [cell("Test"), cell("299 clips", { bold: true })],
], { x: 8.8, y: 1.9, w: 3.5, colW: [1.7, 1.8], rowH: 0.3, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "dataTable" });

s.addText("Test = a whole held-out book (LJ005)", { x: 8.8, y: 4.15, w: 3.5, h: 0.32, fontSize: 11.5, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText("With only one speaker we cannot hold out unseen voices, so we hold out entire source texts instead. No test sentence shares a book with any training sentence.",
  { x: 8.8, y: 4.5, w: 3.5, h: 1.5, fontSize: 10.5, color: INK, isTextBox: true, margin: 0 });
s.addNotes("The control test is the thing to stress: 53.7% is chance, so 94.6% proves the aligner is doing real work. Mention that we validated rather than assumed.");

// ============================ 4 — the models ===============================
pres.addSection({ title: "Models" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Models" });
s.addText("The models we built", { placeholder: "title" });

s.addText("The task:  given a phoneme and its context, predict how many frames it lasts  —  regression, 905,833 training examples",
  { x: 0.7, y: 1.2, w: 11.9, h: 0.34, fontSize: 13, bold: true, color: C.accent3, isTextBox: true, margin: 0 });

s.addText("We TRAINED these three", { x: 0.7, y: 1.68, w: 7.4, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addTable([
  [hdr("Model"), hdr("What it is"), hdr("Test MAE")],
  [cell("Per-phone mean", { bold: true, color: "065A82" }), cell("average length of each phone type — the baseline to beat"), cell(pick(test,"mean","mae_ms").toFixed(2) + " ms", { bold: true })],
  [cell("Ridge regression", { bold: true, color: "065A82" }), cell("linear model, one-hot categories, alpha tuned on validation"), cell(pick(test,"ridge","mae_ms").toFixed(2) + " ms", { bold: true })],
  [cell("Boosted trees", { bold: true, color: "02C39A" }), cell("HistGradientBoosting, native category splits, stopped at 1,052 rounds"), cell(pick(test,"gbt","mae_ms").toFixed(2) + " ms", { bold: true, color: "02C39A" })],
], { x: 0.7, y: 2.02, w: 7.4, colW: [1.75, 4.05, 1.6], rowH: 0.42, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.07, objectName: "modelTable" });

s.addText([{ text: improvement + "% ", options: { fontSize: 20, bold: true, color: "02C39A" } },
           { text: "better than the baseline. Ordering baseline > ridge > trees is the reassuring one —", options: { fontSize: 11.5, color: INK } },
           { text: " if ridge had won we would have gone looking for a bug.", options: { fontSize: 11.5, color: INK, italic: true } }],
  { x: 0.7, y: 3.82, w: 7.4, h: 0.6, isTextBox: true, margin: 0 });

s.addText("21 features, all from the TEXT only", { x: 0.7, y: 4.5, w: 7.4, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addText([
  { text: "the phone itself (with stress), the phone before and after, phone class, position in word and sentence, and distance to the nearest pause", options: { bullet: true, breakLine: true } },
  { text: "we predict log(duration): most phones are ~7 frames but silences reach 481, so raw squared error would chase the pauses", options: { bullet: true, breakLine: true } },
  { text: "NO feature uses the clip's true length — that would leak. A test triples every duration and checks the features are unchanged.", options: { bullet: true } },
], { x: 0.7, y: 4.84, w: 7.4, h: 1.5, fontSize: 11, color: INK, paraSpaceAfter: 5, isTextBox: true, margin: 0 });

card(s, 8.5, 1.68, 4.1, 4.6, "preCard");
s.addText("We USED these, pre-trained", { x: 8.8, y: 1.86, w: 3.5, h: 0.3, fontSize: 12.5, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText("We did not train them — saying otherwise would be wrong.", { x: 8.8, y: 2.18, w: 3.5, h: 0.4, fontSize: 10, italic: true, color: MUTED, isTextBox: true, margin: 0 });
const pre = [
  ["MFA aligner", "finds where each phone starts and ends (our ground truth)"],
  ["HiFi-GAN vocoder", "turns a spectrogram back into a waveform — frozen, so it is identical in every system"],
  ["Whisper small.en", "transcribes our audio so we can score it — the ruler, not the subject"],
];
let cy = 2.66;
for (const [n, d] of pre) {
  s.addText(n, { x: 8.8, y: cy, w: 3.5, h: 0.28, fontSize: 11.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(d, { x: 8.8, y: cy + 0.28, w: 3.5, h: 0.72, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  cy += 1.1;
}
s.addText("Freezing the vocoder is experimental design, not laziness: if it changed between systems the A1 vs A2 comparison would mean nothing.",
  { x: 8.8, y: 5.9, w: 3.5, h: 0.5, fontSize: 9.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("A fourth thing we fitted is the phoneme templates: for each of 70 phones we averaged every training occurrence into one 80x3 matrix. It is a non-parametric acoustic model - fitted from data, just not by gradient descent. Say this if asked what the acoustic model is.");

// ============================ 5 — how we measured ==========================
pres.addSection({ title: "Metrics" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Metrics" });
s.addText("How we measured, and why these metrics", { placeholder: "title" });

const metrics = [
  ["MAE", "Mean Absolute Error", "average of |predicted − actual|, in frames and ms",
   "Why not RMSE? It squares errors, so a few 481-frame silences would dominate. Why not MAPE? It divides by the true value, which explodes on 1-frame phones. Why not R²? Unitless — it cannot say “we are 19.65 ms off”.",
   "duration models", "1C7293"],
  ["WER / CER", "Word / Character Error Rate", "(substitutions + deletions + insertions) ÷ words",
   "Whisper transcribes our audio; we compare to the truth. Why not MCD? It measures spectral distance, which would just say “very different” without telling us if it is understandable. CER is reported too because WER is harsh — one wrong letter kills a whole word.",
   "intelligibility", "065A82"],
  ["MOS", "Mean Opinion Score", "5 listeners × 5 clips, rated 1–5 for naturalness",
   "WER is completely blind to sounding robotic. Audio can be 0% WER and still sound like a 1980s answering machine, so we need a human measure as well.",
   "naturalness", "02C39A"],
  ["Wilcoxon + paired t", "Significance tests", "is the A1 vs A2 difference real, or noise?",
   "Paired, because the same clips go through both systems. Wilcoxon because per-clip WER is skewed, bounded at 0 and full of ties. We report BOTH because they disagree — quoting only the one under 0.05 would be cherry-picking.",
   "both comparisons", "B85042"],
];
let my = 1.28;
for (const [abbr, name, what, why, applies, col] of metrics) {
  card(s, 0.7, my, 11.9, 1.26, "m" + abbr);
  s.addText(abbr, { x: 0.92, y: my + 0.16, w: 1.55, h: 0.4, fontSize: 17, bold: true, color: col, isTextBox: true, margin: 0 });
  s.addText(applies, { x: 0.92, y: my + 0.6, w: 1.55, h: 0.3, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
  s.addText(name, { x: 2.6, y: my + 0.14, w: 3.0, h: 0.3, fontSize: 11.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(what, { x: 2.6, y: my + 0.44, w: 3.0, h: 0.7, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  s.addText(why, { x: 5.85, y: my + 0.16, w: 6.55, h: 0.95, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  my += 1.36;
}
s.addText("We validated the ruler first: Whisper on the REAL recordings scored 1.85% WER. If that had come out at 40%, every other number would be worthless.",
  { x: 0.7, y: 6.8, w: 11.9, h: 0.4, fontSize: 11, bold: true, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addNotes("The training loss and the evaluation metric are deliberately different: we train on squared error in LOG space so long silences do not dominate the fit, but we report MAE in linear frames so the number stays interpretable. Mixing those up is a classic mistake; doing it on purpose is a good answer.");

// ============================ 6 — results ==================================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Results" });
s.addText("Results", { placeholder: "title" });

s.addText("1 — Duration model, on 299 clips from an unseen book", { x: 0.7, y: 1.2, w: 5.9, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addTable([
  [hdr("Model"), hdr("MAE frames"), hdr("MAE ms"), hdr("no silence")],
  [cell("Per-phone mean"), cell(pick(test,"mean","mae_frames").toFixed(3)), cell(pick(test,"mean","mae_ms").toFixed(2)), cell(pick(test,"mean","mae_ms_nosil").toFixed(2))],
  [cell("Ridge"), cell(pick(test,"ridge","mae_frames").toFixed(3)), cell(pick(test,"ridge","mae_ms").toFixed(2)), cell(pick(test,"ridge","mae_ms_nosil").toFixed(2))],
  [cell("Boosted trees", { bold: true, color: "02C39A" }), cell(pick(test,"gbt","mae_frames").toFixed(3), { bold: true, color: "02C39A" }), cell(pick(test,"gbt","mae_ms").toFixed(2), { bold: true, color: "02C39A" }), cell(pick(test,"gbt","mae_ms_nosil").toFixed(2), { bold: true, color: "02C39A" })],
], { x: 0.7, y: 1.54, w: 5.9, colW: [2.0, 1.35, 1.25, 1.3], rowH: 0.36, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "durTable" });

s.addText("Reported twice because silence is only 5.3% of phones but averages 14.6 frames against 7.4 — it would otherwise swamp the headline.",
  { x: 0.7, y: 3.06, w: 5.9, h: 0.5, fontSize: 10, italic: true, color: MUTED, isTextBox: true, margin: 0 });

s.addText("Top features (permutation importance)", { x: 0.7, y: 3.6, w: 5.9, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
const feats = [["phone", impOf("phone")], ["next_phone", impOf("next_phone")], ["prev_phone", impOf("prev_phone")], ["dist_sil_right", impOf("dist_sil_right")]];
const fmax = Math.max(...feats.map((f) => f[1]));
let fy = 3.96;
for (const [n, v] of feats) {
  s.addText(n, { x: 0.7, y: fy, w: 1.65, h: 0.26, fontSize: 10.5, color: INK, isTextBox: true, margin: 0 });
  s.addShape(pres.ShapeType.rect, { x: 2.4, y: fy + 0.05, w: Math.max(0.08, (v / fmax) * 3.0), h: 0.17,
    fill: { color: n === "dist_sil_right" ? "02C39A" : "1C7293" }, objectName: "fb" + n });
  s.addText(v.toFixed(3), { x: 5.5, y: fy, w: 1.0, h: 0.26, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
  fy += 0.32;
}
s.addText("dist_sil_right is 19x more important than dist_sil_left — the model found PRE-pausal lengthening on its own: sounds stretch before an upcoming pause, not after a past one.",
  { x: 0.7, y: 5.3, w: 5.9, h: 0.6, fontSize: 10, italic: true, color: C.accent3, isTextBox: true, margin: 0 });

s.addText("2 — The four systems, 100 clips each", { x: 6.9, y: 1.2, w: 5.7, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addTable([
  [hdr("System"), hdr("What it is"), hdr("WER"), hdr("MOS")],
  [cell("R0", { bold: true, color: "065A82" }), cell("real recording"), cell(f2(wer.R0.wer) + "%", { bold: true }), cell(f2(mos.R0), { bold: true })],
  [cell("R1", { bold: true, color: "065A82" }), cell("real picture → vocoder"), cell(f2(wer.R1.wer) + "%", { bold: true }), cell(f2(mos.R1), { bold: true })],
  [cell("A1", { bold: true, color: "065A82" }), cell("templates + REAL timings"), cell(f2(wer.A1.wer) + "%", { bold: true }), cell(f2(mos.A1), { bold: true })],
  [cell("A2", { bold: true, color: "02C39A" }), cell("templates + PREDICTED"), cell(f2(wer.A2.wer) + "%", { bold: true, color: "02C39A" }), cell(f2(mos.A2), { bold: true, color: "02C39A" })],
], { x: 6.9, y: 1.54, w: 5.7, colW: [0.85, 2.45, 1.2, 1.2], rowH: 0.36, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "sysTable" });

s.addText("3 — Reading the chain: one step at a time", { x: 6.9, y: 3.3, w: 5.7, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
const chain = [
  ["R0 → R1", "+0.63 pp", "the vocoder — nearly free", "1C7293"],
  ["R1 → A1", "+21.17 pp", "averaged templates — ALL the loss", "B85042"],
  ["A1 → A2", "−2.50 pp", "predicting timings — cost nothing", "02C39A"],
];
let chy = 3.66;
for (const [step, cost, what, col] of chain) {
  s.addText(step, { x: 6.9, y: chy, w: 1.1, h: 0.28, fontSize: 10.5, bold: true, color: MUTED, isTextBox: true, margin: 0 });
  s.addText(cost, { x: 7.95, y: chy, w: 1.15, h: 0.28, fontSize: 12.5, bold: true, color: col, isTextBox: true, margin: 0 });
  s.addText(what, { x: 9.15, y: chy + 0.02, w: 3.45, h: 0.28, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  chy += 0.4;
}
card(s, 6.9, 4.95, 5.7, 1.5, "surprise");
s.addText("The surprise", { x: 7.15, y: 5.1, w: 5.2, h: 0.28, fontSize: 11, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText("We expected A2 to be worse. It was not — but A2 wins on only 40 of 100 clips (p = 0.048 and 0.100), and on naturalness 21 of 25 ratings were tied (p = 0.317). So the honest claim is no measurable effect, not that predicted beats real.",
  { x: 7.15, y: 5.38, w: 5.2, h: 1.0, fontSize: 9.5, color: INK, isTextBox: true, margin: 0 });
s.addNotes("Both metrics agree: the duration source is undetectable, the acoustic model is obvious to both. The vocoder costs 0.72 MOS, the templates cost 2.32 MOS.");

// ============================ 7 — what we learned ==========================
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Conclusion" });
s.addText("What we learned", { placeholder: "title" });

const findings = [
  ["Forced alignment gives timing we can trust",
   "94.6% of boundaries verified against a 53.7% control, all 13,100 clips aligned, none excluded. It reproduced phonetics we never encoded: stressed vowels 108 ms vs 67 ms."],
  ["Context features clearly beat a per-phone average",
   improvement + "% lower duration error (" + pick(test,"mean","mae_ms").toFixed(2) + " ms → " + pick(test,"gbt","mae_ms").toFixed(2) + " ms). The model independently recovered pre-pausal lengthening."],
  ["We set out to measure one thing and learned the other matters ~30x more",
   "Predicting durations cost nothing measurable on EITHER metric. The crude acoustic model cost 21 points of WER and 2.32 MOS. So the next effort belongs in the acoustic model — exactly where a diffusion generator would sit."],
];
let y = 1.35;
for (const [h, d] of findings) {
  s.addShape(pres.ShapeType.ellipse, { x: 0.75, y: y + 0.04, w: 0.3, h: 0.3, fill: { color: C.accent2 }, objectName: "fd" + y });
  s.addText(h, { x: 1.2, y: y, w: 11.3, h: 0.36, fontSize: 15, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText(d, { x: 1.2, y: y + 0.38, w: 11.3, h: 0.62, fontSize: 11.5, color: "AEC6D2", isTextBox: true, margin: 0 });
  y += 1.22;
}
s.addText("Being honest about the limits", { x: 0.75, y: 5.1, w: 11.5, h: 0.3, fontSize: 12.5, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText([
  { text: "A2 is told the clip's true total length, so it is not yet a fully autonomous system", options: { bullet: true, breakLine: true } },
  { text: "The listening study is 5 team members, not blind naive listeners — it corroborates WER rather than measuring naturalness independently", options: { bullet: true, breakLine: true } },
  { text: "One speaker only, and a deliberately crude acoustic model that caps how good A1 and A2 can ever sound", options: { bullet: true } },
], { x: 0.75, y: 5.44, w: 11.5, h: 1.1, fontSize: 11, color: "8FA9B8", paraSpaceAfter: 4, isTextBox: true, margin: 0 });
s.addText("Next: predict total length too  ·  replace templates with a learned acoustic model  ·  repeat the listening study blind",
  { x: 0.75, y: 6.72, w: 11.5, h: 0.35, fontSize: 11, italic: true, color: "02C39A", isTextBox: true, margin: 0 });
s.addNotes("If pushed on why the synthetic audio is worse than the real recording: that is the expected consequence of a deliberately simple acoustic model. The project was never trying to win on audio quality - the experiment is the A1 vs A2 comparison, which is only meaningful because the acoustic model is held fixed.");

// ---- write ------------------------------------------------------------------
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
